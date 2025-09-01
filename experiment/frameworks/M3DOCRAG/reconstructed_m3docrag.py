import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# from transformers import AutoProcessor # Not directly used after GmeQwen2VL initialization
from gme_inference import GmeQwen2VL

from PIL import Image, ImageFile
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import os
import shutil
import traceback
from openai import OpenAI
import pickle
import glob
import json 
from datetime import datetime 
from concurrent.futures import ThreadPoolExecutor, as_completed # Added for concurrency

ImageFile.LOAD_TRUNCATED_IMAGES = True 

OPENAI_API_KEY="your-api-key"  # Default API key
# Ensure the API key is available in the environment for the OpenAI library
if "OPENAI_API_KEY" not in os.environ and OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataclass changes ---
@dataclass
class PageInfo:
    doc_id: str
    page_num_in_doc: int
    page_image_path: str 

@dataclass
class DocumentPage:
    doc_id: str
    page_num: int
    image: Image.Image = field(repr=False)
    page_image_path: str 

@dataclass
class RetrievedPage:
    page: DocumentPage
    score: float

@dataclass
class RetrievedPageInfoForEval:
    doc_id: str
    page_num_in_doc: int
    page_image_path: str 
    score: float
    retrieved_index: int

class SimplifiedM3DOCRAG:
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
        max_pages_to_retrieve: int = 4, 
        use_approximate_index: bool = True,
        batch_size: int = 4, # Batch size for embedding
        index_storage_path: str = "path/to/your/storage",
        pages_per_embedding_chunk: int = 200,
        samples_for_faiss_training: int = 5000,
        force_rebuild_index: bool = False
    ):
        self.openai_api_key = openai_api_key
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required.")
        self.model_name = model_name
        self.max_pages_to_retrieve = max_pages_to_retrieve
        self.use_approximate_index = use_approximate_index
        self.batch_size = batch_size
        self.pages_per_embedding_chunk = pages_per_embedding_chunk
        self.samples_for_faiss_training = samples_for_faiss_training
        self._force_rebuild_index_default = force_rebuild

        self.index_storage_path = index_storage_path
        os.makedirs(self.index_storage_path, exist_ok=True)
        self.faiss_index_path = os.path.join(self.index_storage_path, "path/to/your/doc.index")
        self.page_infos_path = os.path.join(self.index_storage_path, "path/to/your/infos.pkl")
        self.embedding_checkpoints_dir = os.path.join(self.index_storage_path, "embedding_checkpoints")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.retrieval_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using retrieval device: {self.retrieval_device}")

        logging.info(f"Initializing retrieval model: {model_name}...")
        
        self.model = GmeQwen2VL(model_name=model_name, # Use passed model_name
                                model_path=r'path/to/your/model',
                                device=self.retrieval_device)
        
        # Determine image size for dummy images if needed
        self.retrieval_processor_image_size = 224 # Default
        try:
            if hasattr(self.model, 'processor') and self.model.processor and \
               hasattr(self.model.processor, 'image_config') and 'image_size' in self.model.processor.image_config:
                 self.retrieval_processor_image_size = self.model.processor.image_config['image_size']
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'config') and \
                 hasattr(self.model.model.config, 'vision_config') and hasattr(self.model.model.config.vision_config, 'image_size'):
                 self.retrieval_processor_image_size = self.model.model.config.vision_config.image_size
            # Add more specific checks if GmeQwen2VL has a direct way to get this
        except Exception as e:
            logging.warning(f"Could not reliably determine retrieval_processor.image_size from model. Using default {self.retrieval_processor_image_size}. Error: {e}")


        self.index: Optional[faiss.Index] = None
        self.page_infos_list: List[PageInfo] = []
        logging.info("Models initialized successfully!")

    def _get_document_subfolder_paths(self, corpus_root_path: str) -> List[str]:
        doc_subfolder_paths = []
        if not os.path.isdir(corpus_root_path):
            logging.error(f"Corpus root path {corpus_root_path} does not exist or is not a directory.")
            return doc_subfolder_paths
        for item_name in os.listdir(corpus_root_path):
            item_path = os.path.join(corpus_root_path, item_name)
            if os.path.isdir(item_path):
                doc_subfolder_paths.append(item_path)
        logging.info(f"Found {len(doc_subfolder_paths)} potential document subfolders in {corpus_root_path}.")
        return doc_subfolder_paths

    def _collect_page_infos(self, doc_subfolder_paths: List[str]) -> List[PageInfo]:
        all_page_infos: List[PageInfo] = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        
        for doc_path in tqdm(doc_subfolder_paths, desc="Scanning document folders for page images"):
            doc_id = os.path.basename(doc_path)
            try:
                image_files = []
                for ext in supported_extensions:
                    image_files.extend(glob.glob(os.path.join(doc_path, f"*{ext}")))
                image_files.extend(glob.glob(os.path.join(doc_path, f"*{ext.upper()}"))) # Also check uppercase

                def sort_key(filepath):
                    try:
                        return int(os.path.splitext(os.path.basename(filepath))[0])
                    except ValueError: # Handle non-numeric filenames gracefully for sorting
                        return os.path.splitext(os.path.basename(filepath))[0] 
                image_files.sort(key=sort_key)

                for img_path in image_files:
                    try:
                        page_num_str = os.path.splitext(os.path.basename(img_path))[0]
                        page_num = int(page_num_str) # Assumes filename is page number
                        all_page_infos.append(PageInfo(doc_id, page_num, img_path))
                    except ValueError:
                        logging.warning(f"Could not parse page number from filename {img_path} in {doc_id}. Skipping.")
                    except Exception as e_file:
                        logging.warning(f"Error processing file {img_path} in {doc_id}: {e_file}. Skipping.")
            except Exception as e_doc:
                logging.warning(f"Error scanning document folder {doc_path}: {e_doc}. Skipping.")
        return all_page_infos

    def _load_images_for_page_infos(self, current_page_infos: List[PageInfo]) -> List[Image.Image]:
        images = []
        for pi in tqdm(current_page_infos, desc="Loading images for chunk", leave=False, disable=len(current_page_infos)<10):
            try:
                page_image = Image.open(pi.page_image_path).convert("RGB")
                images.append(page_image)
            except FileNotFoundError:
                logging.warning(f"Image file not found: {pi.page_image_path} (doc {pi.doc_id}, page {pi.page_num_in_doc}). Using dummy.")
                images.append(Image.new("RGB", (self.retrieval_processor_image_size, self.retrieval_processor_image_size), (220,220,220)))
            except Exception as e:
                logging.warning(f"Error loading image {pi.page_image_path} (doc {pi.doc_id}, page {pi.page_num_in_doc}): {e}. Using dummy.")
                images.append(Image.new("RGB", (self.retrieval_processor_image_size, self.retrieval_processor_image_size), (220,220,220)))
        return images

    def _get_embeddings_for_images(self, images_to_embed: List[Image.Image]) -> Optional[np.ndarray]:
        if not images_to_embed:
            return None
        
        embeddings_doc_all: Optional[torch.Tensor] = None
        try:
            if self.retrieval_device.startswith("cuda"):
                with torch.cuda.device(self.retrieval_device):
                    torch.cuda.empty_cache()
                    embeddings_doc_all = self.model.get_image_embeddings(images=images_to_embed, batch_size=self.batch_size)
            else:
                embeddings_doc_all = self.model.get_image_embeddings(images=images_to_embed, batch_size=self.batch_size)

            if embeddings_doc_all is None:
                logging.error("Model returned None for image embeddings.")
                return None

            if embeddings_doc_all.ndim == 3 and embeddings_doc_all.shape[1] > 1: 
                embeddings_doc_all = embeddings_doc_all.mean(dim=1) 
            
            final_embeddings_tensor = embeddings_doc_all.to(dtype=torch.float32).cpu()
            return final_embeddings_tensor.numpy()
        except Exception as e:
            logging.error(f"Error during image embedding: {e}", exc_info=True)
            return None


    def _ensure_embedding_checkpoints(self, page_infos: List[PageInfo], checkpoints_dir: str) -> List[str]:
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_file_paths = []
        num_chunks = (len(page_infos) + self.pages_per_embedding_chunk - 1) // self.pages_per_embedding_chunk
        for i in tqdm(range(num_chunks), desc="Ensuring embedding checkpoints"):
            chunk_start, chunk_end = i * self.pages_per_embedding_chunk, min((i + 1) * self.pages_per_embedding_chunk, len(page_infos))
            cp_file = os.path.join(checkpoints_dir, f"chunk_{i:05d}.npy")
            checkpoint_file_paths.append(cp_file)
            
            if os.path.exists(cp_file) and os.path.getsize(cp_file) > 0: # Check if file exists and is not empty
                 try: # Verify if it's a valid numpy file with expected structure
                    data = np.load(cp_file)
                    if data.ndim == 2 and data.shape[0] == (chunk_end - chunk_start) : # Simple check
                        logging.debug(f"Checkpoint {cp_file} exists and seems valid. Skipping regeneration.")
                        continue
                    else:
                         logging.warning(f"Checkpoint {cp_file} exists but seems invalid or incomplete. Will regenerate.")
                 except Exception as e_load_check:
                     logging.warning(f"Checkpoint {cp_file} exists but failed to load/validate ({e_load_check}). Will regenerate.")


            infos_chunk = page_infos[chunk_start:chunk_end]
            if not infos_chunk: 
                np.save(cp_file, np.array([])); 
                continue
            
            images_chunk = self._load_images_for_page_infos(infos_chunk) 
            if not images_chunk: # All images in chunk failed to load
                np.save(cp_file, np.array([]));
                del images_chunk
                continue

            embeddings_np = self._get_embeddings_for_images(images_chunk)
            del images_chunk # Free memory
            
            if embeddings_np is not None and embeddings_np.size > 0:
                np.save(cp_file, embeddings_np)
            else: # Handle cases where embedding failed or returned empty
                logging.warning(f"Embeddings for chunk {i} were None or empty. Saving empty array to {cp_file}.")
                np.save(cp_file, np.array([])) # Save an empty array to mark as processed but failed
        return checkpoint_file_paths

    def _get_embedding_dim_from_checkpoints(self, checkpoint_paths: List[str]) -> int:
        for cp_path in checkpoint_paths:
            if not os.path.exists(cp_path) or os.path.getsize(cp_path) == 0: continue
            try:
                embeddings = np.load(cp_path)
                if embeddings.ndim == 2 and embeddings.shape[0] > 0: return embeddings.shape[1]
            except Exception as e: 
                logging.warning(f"Could not load or parse checkpoint {cp_path} for dim check: {e}")
                continue
        
        logging.warning("Could not get embedding dim from checkpoints. Trying model config/dummy image.")
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config') and hasattr(self.model.model.config, 'hidden_size'):
                logging.info(f"Using embedding dim from model.model.config.hidden_size: {self.model.model.config.hidden_size}")
                return self.model.model.config.hidden_size
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'): 
                logging.info(f"Using embedding dim from model.config.hidden_size: {self.model.config.hidden_size}")
                return self.model.config.hidden_size
        except AttributeError: pass # Expected if attributes don't exist

        dummy_img = Image.new("RGB", (self.retrieval_processor_image_size, self.retrieval_processor_image_size), (255,255,255))
        embeddings_np = self._get_embeddings_for_images([dummy_img])
        if embeddings_np is not None and embeddings_np.ndim ==2 and embeddings_np.shape[0] > 0: 
            logging.info(f"Using embedding dim from dummy image: {embeddings_np.shape[1]}")
            return embeddings_np.shape[1]
        
        logging.error("CRITICAL: Failed to determine embedding dim. Defaulting to 4096 (common for Qwen-VL)."); 
        return 4096 

    def _collect_training_embeddings_from_checkpoints(self, checkpoint_paths: List[str], num_samples: int, emb_dim: int) -> Optional[np.ndarray]:
        training_embs_list = []
        count = 0
        
        # Shuffle checkpoints to get a more diverse sample if num_samples is small
        import random
        shuffled_checkpoint_paths = random.sample(checkpoint_paths, len(checkpoint_paths))

        for cp_path in tqdm(shuffled_checkpoint_paths, desc="Loading checkpoints for FAISS training", leave=False, disable=len(shuffled_checkpoint_paths)<5):
            if count >= num_samples: break
            if not os.path.exists(cp_path) or os.path.getsize(cp_path) == 0: continue # Skip empty or non-existent
            try:
                chunk_embs = np.load(cp_path)
                if chunk_embs.ndim == 2 and chunk_embs.shape[0] > 0 and chunk_embs.shape[1] == emb_dim:
                    take = min(num_samples - count, chunk_embs.shape[0])
                    indices_to_take = np.random.choice(chunk_embs.shape[0], take, replace=False) if take < chunk_embs.shape[0] else np.arange(chunk_embs.shape[0])
                    training_embs_list.append(chunk_embs[indices_to_take])
                    count += take
                elif chunk_embs.size == 0: # Empty array from a failed chunk
                    logging.debug(f"Skipping empty embedding array from {cp_path} for training.")
                elif chunk_embs.shape[1] != emb_dim:
                     logging.warning(f"Embedding dim mismatch in {cp_path} (expected {emb_dim}, got {chunk_embs.shape[1]}). Skipping for training.")
            except Exception as e: 
                logging.error(f"Error loading or processing checkpoint {cp_path} for FAISS training: {e}. Skipping.")
                continue
        
        if not training_embs_list: 
            logging.warning("No training embeddings collected for FAISS.")
            return None
        
        try:
            final_training_embs = np.concatenate(training_embs_list, axis=0)
            if final_training_embs.shape[0] == 0:
                logging.warning("Concatenated training embeddings are empty.")
                return None
            return final_training_embs
        except ValueError as e_concat: # Handle empty list causing concatenate error
            logging.error(f"Error concatenating training embeddings: {e_concat}")
            return None


    def build_or_load_index(self, current_force_rebuild: bool, corpus_root_path: Optional[str] = None): 
        if current_force_rebuild:
            logging.warning("Force rebuild initiated by parameter.")
            user_confirmation = input("ARE YOU SURE you want to force rebuild? This will delete existing index, page infos, and checkpoints. (yes/no): ")
            if user_confirmation.lower() != 'yes':
                logging.info("Force rebuild cancelled by user.")
                current_force_rebuild = False # Do not force rebuild
            else:
                logging.info("Force rebuild confirmed: Clearing old index, page infos, and embedding checkpoints.")
                if os.path.exists(self.embedding_checkpoints_dir): shutil.rmtree(self.embedding_checkpoints_dir)
                if os.path.exists(self.faiss_index_path): os.remove(self.faiss_index_path)
                if os.path.exists(self.page_infos_path): os.remove(self.page_infos_path)
        
        os.makedirs(self.embedding_checkpoints_dir, exist_ok=True)

        # Determine if index needs to be built
        needs_building = current_force_rebuild or \
                         not os.path.exists(self.faiss_index_path) or \
                         not os.path.exists(self.page_infos_path) or \
                         os.path.getsize(self.faiss_index_path) == 0 # Also rebuild if index file is empty

        if needs_building:
            if not corpus_root_path: 
                raise ValueError("Corpus root path is required to build a new index.") 
            logging.info(f"Building new index from images in: {corpus_root_path}") 
            
            # Load or generate page_infos_list
            if not current_force_rebuild and os.path.exists(self.page_infos_path) and os.path.getsize(self.page_infos_path) > 0:
                try:
                    with open(self.page_infos_path, "rb") as f: self.page_infos_list = pickle.load(f)
                    logging.info(f"Loaded existing page_infos_list with {len(self.page_infos_list)} items for new index build.")
                except Exception as e:
                    logging.warning(f"Could not load existing page_infos_path {self.page_infos_path}, will regenerate: {e}")
                    self.page_infos_list = [] 
            
            if not self.page_infos_list or current_force_rebuild : 
                doc_subfolder_paths = self._get_document_subfolder_paths(corpus_root_path)
                if not doc_subfolder_paths:
                    logging.warning(f"No document subfolders found in {corpus_root_path}. Index will be empty.")
                    self.page_infos_list = []
                else:
                    self.page_infos_list = self._collect_page_infos(doc_subfolder_paths)
                
                try: # Save even if empty to signify processing attempt
                    with open(self.page_infos_path, "wb") as f: pickle.dump(self.page_infos_list, f)
                except Exception as e:
                    logging.error(f"Failed to save page_infos_list to {self.page_infos_path}: {e}")

            logging.info(f"Total pages to process for index: {len(self.page_infos_list)}.")
            if not self.page_infos_list: 
                logging.warning("No page images found to index. Creating an empty index file.")
                emb_dim_for_empty = self._get_embedding_dim_from_checkpoints([]) 
                self.index = faiss.IndexFlatIP(emb_dim_for_empty) 
                faiss.write_index(self.index, self.faiss_index_path)
                logging.info(f"Empty index created at {self.faiss_index_path} and empty page_infos at {self.page_infos_path}")
                return

            checkpoint_paths = self._ensure_embedding_checkpoints(self.page_infos_list, self.embedding_checkpoints_dir)
            
            has_valid_embeddings_in_checkpoints = False
            for fp in checkpoint_paths:
                if os.path.exists(fp) and os.path.getsize(fp) > 0:
                    try:
                        data = np.load(fp)
                        if data.ndim == 2 and data.shape[0] > 0: 
                            has_valid_embeddings_in_checkpoints = True; break
                    except Exception: pass # Ignore load errors here, dim check will handle later
            
            if not has_valid_embeddings_in_checkpoints:
                logging.warning("No valid embeddings found in checkpoints after processing. Creating an empty index.")
                emb_dim_for_empty = self._get_embedding_dim_from_checkpoints(checkpoint_paths) # Try to get dim anyway
                self.index = faiss.IndexFlatIP(emb_dim_for_empty)
                faiss.write_index(self.index, self.faiss_index_path)
                return

            emb_dim = self._get_embedding_dim_from_checkpoints(checkpoint_paths)
            
            # Count total vectors accurately from valid checkpoints
            total_vecs = 0
            valid_checkpoint_embeddings_for_adding = []
            for fp in checkpoint_paths:
                if os.path.exists(fp) and os.path.getsize(fp) > 0:
                    try:
                        data = np.load(fp)
                        if data.ndim == 2 and data.shape[0] > 0 and data.shape[1] == emb_dim:
                            total_vecs += data.shape[0]
                            valid_checkpoint_embeddings_for_adding.append(data) # Store for adding later
                        elif data.size > 0 : # Exists but is not valid (e.g. wrong dim)
                             logging.warning(f"Checkpoint {fp} has data but is not valid for adding to index (dim: {data.shape}, expected dim: {emb_dim}). Skipping.")
                    except Exception as e_loadadd: 
                        logging.warning(f"Could not load embeddings from {fp} for adding to index: {e_loadadd}")
            
            if total_vecs == 0:
                logging.warning("No valid vectors to add to the FAISS index after processing all checkpoints. Creating an empty index.")
                self.index = faiss.IndexFlatIP(emb_dim)
                faiss.write_index(self.index, self.faiss_index_path)
                return

            use_approx = self.use_approximate_index and total_vecs >= 156 # Min samples for IVF training often suggested higher

            if use_approx:
                logging.info(f"Attempting to build approximate IVF index. Total vectors: {total_vecs}")
                # Use a portion of all valid embeddings for training if samples_for_faiss_training is large
                # Or use dedicated _collect_training_embeddings_from_checkpoints if distinct sample needed
                train_embs_np_arr = self._collect_training_embeddings_from_checkpoints(checkpoint_paths, self.samples_for_faiss_training, emb_dim)

                if train_embs_np_arr is not None and train_embs_np_arr.shape[0] > 0:
                    n_train = train_embs_np_arr.shape[0]
                    # n_centroids: Faiss recommends 4*sqrt(N) to 16*sqrt(N). For training, N is n_train.
                    # Also, ensure at least 39 training samples per centroid (min_points_per_centroid).
                    min_points_per_centroid = 39 
                    # Max centroids to avoid excessive memory/slowness, e.g., 1024 or 4096 for very large datasets
                    # Here using a simpler cap like 256 or based on n_train.
                    n_centroids_ideal = max(1, int(4 * np.sqrt(n_train))) 
                    n_centroids_max_from_samples = max(1, n_train // min_points_per_centroid if n_train >= min_points_per_centroid else 1)
                    n_centroids = min(n_centroids_ideal, n_centroids_max_from_samples, 256) # Cap at 256
                    if n_centroids == 0 : n_centroids = 1 # Must be at least 1

                    logging.info(f"Training IndexIVFFlat with {n_train} samples for {n_centroids} centroids (dim={emb_dim}).")
                    quantizer = faiss.IndexFlatIP(emb_dim) # Inner product for cosine similarity after normalization
                    try:
                        self.index = faiss.IndexIVFFlat(quantizer, emb_dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
                        self.index.train(train_embs_np_arr)
                    except Exception as e_train:
                        logging.error(f"Failed to train IndexIVFFlat: {e_train}. Falling back to IndexFlatIP.")
                        self.index = None 
                        use_approx = False # Fallback
                else: 
                    use_approx = False
                    logging.info("Not enough training samples or error collecting them for approximate index. Falling back to IndexFlatIP.")
            
            if not use_approx or self.index is None or not isinstance(self.index, faiss.IndexIVFFlat): 
                logging.info(f"Using IndexFlatIP (dim={emb_dim}).")
                self.index = faiss.IndexFlatIP(emb_dim)

            # Add embeddings from valid_checkpoint_embeddings_for_adding
            logging.info(f"Populating FAISS index with {total_vecs} vectors...")
            for embs_to_add in tqdm(valid_checkpoint_embeddings_for_adding, desc="Adding embeddings to FAISS"):
                faiss.normalize_L2(embs_to_add) 
                self.index.add(embs_to_add)
            
            faiss.write_index(self.index, self.faiss_index_path)
            logging.info(f"Index built with {self.index.ntotal} vectors and saved to {self.faiss_index_path}")
        
        else: # Load existing index and page_infos
            try:
                logging.info(f"Loading existing index from {self.faiss_index_path}")
                self.index = faiss.read_index(self.faiss_index_path)
                logging.info(f"Loading existing page infos from {self.page_infos_path}")
                with open(self.page_infos_path, "rb") as f: self.page_infos_list = pickle.load(f)
                logging.info(f"Loaded index ({self.index.ntotal} vectors) & page infos ({len(self.page_infos_list)} pages).")
                
                # Sanity checks
                if self.index.ntotal == 0 and len(self.page_infos_list) > 0 :
                    logging.warning("Loaded FAISS index is empty but page_infos exist. Index may be corrupted or wasn't built correctly. Consider rebuilding.")
                elif self.index.ntotal > 0 and len(self.page_infos_list) == 0:
                    logging.warning("Loaded FAISS index has vectors but page_infos list is empty. This is inconsistent. Consider rebuilding.")
                elif self.index.ntotal != len(self.page_infos_list) and self.index.ntotal > 0 and len(self.page_infos_list) > 0 : 
                    logging.warning(f"Mismatch: FAISS index has {self.index.ntotal} vectors, but {len(self.page_infos_list)} page_infos. This might be okay if some pages failed embedding, or problematic if files were moved/deleted. Consider rebuilding if issues arise.")

            except Exception as e:
                logging.error(f"Error loading existing index or page_infos: {e}. Clearing potentially corrupted data. You may need to rebuild the index.", exc_info=True); 
                self.index = None
                self.page_infos_list = []


    def _load_image_for_page_info(self, page_info: PageInfo) -> Image.Image:
        try:
            page_image = Image.open(page_info.page_image_path).convert("RGB")
            return page_image
        except FileNotFoundError:
            logging.error(f"IMAGE LOAD FAIL (Not Found): {page_info.page_image_path} for doc {page_info.doc_id} pg {page_info.page_num_in_doc}")
        except Exception as e:
            logging.error(f"IMAGE LOAD FAIL (Other Error): {page_info.page_image_path} for doc {page_info.doc_id} pg {page_info.page_num_in_doc}: {e}")
        
        # Fallback error image
        error_img = Image.new("RGB", (self.retrieval_processor_image_size, self.retrieval_processor_image_size), (255,0,0)) # Bright red
        try: 
            from PIL import ImageDraw
            draw = ImageDraw.Draw(error_img)
            draw.text((10, 10), "LOAD FAILED", fill=(255,255,255)) # White text
        except ImportError: pass # Pillow might not be fully available in some edge cases
        return error_img

    def _encode_image_to_base64(self, image: Image.Image) -> str: 
        import base64
        from io import BytesIO
        buffered = BytesIO()
        try:
            image.save(buffered, format="PNG") # PNG is lossless and widely supported
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e_save:
            logging.error(f"Failed to save image to buffer for base64 encoding: {e_save}")
            # Fallback: try JPEG if PNG fails for some reason (e.g. specific image mode issues)
            try:
                image.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e_save_jpeg:
                logging.error(f"Failed to save image as JPEG for base64 encoding: {e_save_jpeg}")
                raise # Re-raise if both fail


    def _search_index_for_query(self, query: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.index is None or self.index.ntotal == 0: 
            logging.warning("Search attempted on uninitialized or empty FAISS index.")
            return None, None

        with torch.no_grad():
            query_embedding_full: Optional[torch.Tensor] = None
            try:
                if self.retrieval_device.startswith("cuda"):
                    with torch.cuda.device(self.retrieval_device): 
                        torch.cuda.empty_cache()
                        query_embedding_full = self.model.get_text_embeddings([query], batch_size=1)
                else: 
                    query_embedding_full = self.model.get_text_embeddings([query], batch_size=1)
                
                if query_embedding_full is None:
                    logging.error("Text embedding for query returned None.")
                    return None, None

                if query_embedding_full.ndim == 3 and query_embedding_full.shape[1] > 1: 
                    query_embedding = query_embedding_full.mean(dim=1) 
                else: 
                    query_embedding = query_embedding_full
                
                query_embedding_np = query_embedding.to(dtype=torch.float32).cpu().numpy()
                faiss.normalize_L2(query_embedding_np) 
            except Exception as e_embed_query:
                logging.error(f"Error embedding query '{query[:50]}...': {e_embed_query}", exc_info=True)
                return None, None


        if hasattr(self.index, 'nprobe') and isinstance(self.index, faiss.IndexIVF):
            n_centroids = self.index.invlists.nlist if hasattr(self.index.invlists, 'nlist') else 1
            # Default nprobe: Faiss default is 1. Higher values (e.g., 8, 16, sqrt(n_centroids)) improve recall at cost of speed.
            # Let's use a modest default if n_centroids is large enough, e.g., 10% or a fixed value.
            if n_centroids > 1:
                # A common heuristic is sqrt(n_centroids), or a fixed small number like 10-20.
                # Ensure nprobe is at least 1 and <= n_centroids.
                self.index.nprobe = min(max(1, int(np.sqrt(n_centroids))), n_centroids) 
                # self.index.nprobe = min(max(1, n_centroids // 10 if n_centroids >=10 else 1), 32) # Alternative: 10% capped at 32
                if self.index.nprobe == 0 and n_centroids > 0 : self.index.nprobe = 1 # safety
            else:
                 self.index.nprobe = 1 # Default for non-IVF or single centroid
            logging.debug(f"Using nprobe: {self.index.nprobe} for IVF search (n_centroids={n_centroids}).")
        
        try:
            return self.index.search(query_embedding_np, self.max_pages_to_retrieve)
        except Exception as e_search:
            logging.error(f"Error during FAISS search for query '{query[:50]}...': {e_search}", exc_info=True)
            return None, None

    def retrieve_pages(self, query: str) -> List[RetrievedPage]: 
        if not self.page_infos_list: 
            logging.warning("Attempted to retrieve pages but no page_infos_list available (index might be empty or not loaded).")
            return []
        
        scores_arr, indices_arr = self._search_index_for_query(query)
        if scores_arr is None or indices_arr is None: 
            logging.warning(f"Search returned no results for query: {query[:50]}...")
            return []
        
        retrieved_pages_results: List[RetrievedPage] = []
        valid_indices_found = 0
        for score, idx in zip(scores_arr[0], indices_arr[0]):
            if idx == -1 : continue # FAISS uses -1 for no result in that slot
            if idx >= len(self.page_infos_list):
                logging.warning(f"Retrieved index {idx} is out of bounds for page_infos_list (len {len(self.page_infos_list)}). Skipping.")
                continue
            
            valid_indices_found +=1
            page_info = self.page_infos_list[idx]
            image = self._load_image_for_page_info(page_info) 
            doc_page = DocumentPage(
                doc_id=page_info.doc_id, 
                page_num=page_info.page_num_in_doc, 
                image=image, 
                page_image_path=page_info.page_image_path 
            )
            retrieved_pages_results.append(RetrievedPage(page=doc_page, score=float(score)))
        
        if valid_indices_found == 0 and len(indices_arr[0]) > 0:
             logging.info(f"No valid page indices found from search results for query: {query[:50]}...")
        return retrieved_pages_results

    def retrieve_page_infos_for_evaluation(self, query: str) -> List[RetrievedPageInfoForEval]: 
        # This method is similar to retrieve_pages but returns lighter PageInfo objects, used if images aren't needed immediately.
        # For the current concurrent QA, retrieve_pages (which loads images) is used directly.
        # This method can be kept for other evaluation scenarios if needed.
        if not self.page_infos_list: 
            logging.warning("Attempted to retrieve page infos for eval but no page_infos_list available.")
            return []
        scores_arr, indices_arr = self._search_index_for_query(query)
        if scores_arr is None or indices_arr is None: return []
        
        retrieved_results: List[RetrievedPageInfoForEval] = []
        for score, original_idx in zip(scores_arr[0], indices_arr[0]):
            if original_idx == -1 or original_idx >= len(self.page_infos_list): continue
            page_info = self.page_infos_list[original_idx]
            retrieved_results.append(RetrievedPageInfoForEval(
                doc_id=page_info.doc_id, 
                page_num_in_doc=page_info.page_num_in_doc, 
                page_image_path=page_info.page_image_path, 
                score=float(score), 
                retrieved_index=int(original_idx)
            ))
        return retrieved_results

    def answer_query_with_openai(self, query: str, retrieved_pages: List[RetrievedPage]) -> str: 
        no_info_response = "I cannot find this information in the provided document pages."
        if not retrieved_pages: 
            return no_info_response
        
        system_prompt_content = """You are a document analyzer that ONLY gives two types of responses:
1. If you find the EXACT information: Respond with ONLY that specific information
2. If you cannot find the EXACT information: Respond with EXACTLY and ONLY this phrase: "I cannot find this information in the provided document pages."
3. Answer in the same language as the query.

DO NOT:
- Explain your limitations
- Talk about AI or models
- Make assumptions
- Give partial information
- Provide multiple answers
- Add any explanations or conversational filler"""
        
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": f"Query: {query}"}]
        images_added_count = 0
        for rp_idx, rp in enumerate(retrieved_pages): 
            if rp.page.image is None: # Should have been handled by _load_image_for_page_info creating a dummy
                logging.warning(f"Retrieved page image is None for doc {rp.page.doc_id} pg {rp.page.page_num}. Skipping for OpenAI.")
                continue
            try:
                # Check if it's the dummy error image (e.g., by a specific color or attribute if set)
                # This is a simple check; a more robust way would be to flag failed loads earlier.
                if rp.page.image.mode == "RGB" and rp.page.image.size == (self.retrieval_processor_image_size, self.retrieval_processor_image_size) and \
                   rp.page.image.getpixel((0,0)) == (255,0,0): # Check if it's the red error image
                    logging.warning(f"Skipping dummy error image for OpenAI: {rp.page.page_image_path}")
                    continue

                b64_image = self._encode_image_to_base64(rp.page.image)
                user_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                        "detail": "low" 
                    }
                })
                images_added_count += 1
            except Exception as e_encode:
                logging.warning(f"Could not encode image {rp.page.page_image_path} for doc {rp.page.doc_id} pg {rp.page.page_num} for OpenAI: {e_encode}. Skipping this image.")
        
        if images_added_count == 0: # Only text query, no valid images successfully encoded/added
             logging.warning(f"No valid images were available or successfully encoded for query '{query[:50]}...'. OpenAI will not receive images.")
             return f"{no_info_response} (No usable document page images were provided for analysis)"


        messages = [
            {"role": "system", "content": system_prompt_content}, 
            {"role": "user", "content": user_content}
        ]
        try:
            # Using gpt-4o as specified
            response = self.openai_client.chat.completions.create(
                model="gpt-4o", 
                messages=messages, 
                max_tokens=512, 
                temperature=0.0 # Lower temperature for more deterministic factual extraction
            )
            return response.choices[0].message.content.strip()
        except Exception as e: 
            logging.error(f"Error during OpenAI API call for query '{query[:50]}...': {e}", exc_info=True); 
            return f"Error calling OpenAI API: {str(e)}"

    # --- New method for concurrent OpenAI calls ---
    def answer_queries_concurrently(self, queries_with_pages: List[Tuple[str, List[RetrievedPage]]], num_workers: int = 5) -> List[str]:
        answers = [""] * len(queries_with_pages) 
        future_to_index: Dict[Any, int] = {}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for index, (query, pages) in enumerate(queries_with_pages):
                if not query: # Handle empty query case if it slips through
                    answers[index] = "Error: Empty query provided."
                    continue
                future = executor.submit(self.answer_query_with_openai, query, pages)
                future_to_index[future] = index
            
            # Create a list of futures that were actually submitted
            active_futures = [f for f in future_to_index.keys()] # future_to_index contains only active ones

            if not active_futures:
                logging.info("No queries submitted for concurrent OpenAI processing.")
                return answers # Return pre-allocated empty strings or error messages

            for future in tqdm(as_completed(active_futures), total=len(active_futures), desc="Stage 2: Generating answers (OpenAI concurrent)"):
                original_index = future_to_index[future]
                try:
                    answer = future.result()
                    answers[original_index] = answer
                except Exception as e:
                    logging.error(f"Exception retrieving result for query index {original_index} (query: '{queries_with_pages[original_index][0][:30]}...'): {e}", exc_info=True)
                    answers[original_index] = f"System error during concurrent processing for this query: {str(e)}" # Fallback error
        return answers

    def process_query(self, query: str, corpus_root_path: Optional[str] = None, force_rebuild: Optional[bool] = None) -> str: 
        current_force_rebuild = self._force_rebuild_index_default if force_rebuild is None else force_rebuild
        try:
            self.build_or_load_index(current_force_rebuild=current_force_rebuild, corpus_root_path=corpus_root_path) 
        except ValueError as ve: 
            logging.error(f"Failed to initialize index for process_query: {ve}")
            return f"System error: Failed to initialize index. {ve}"
        except Exception as e_build:
            logging.error(f"Unexpected error during index build/load for process_query: {e_build}", exc_info=True)
            return f"System error: Unexpected error during index build/load."

        if self.index is None : 
             logging.error("System error: Index not properly initialized for process_query.")
             if len(self.page_infos_list) > 0 and (not os.path.exists(self.faiss_index_path) or (hasattr(self.index, 'ntotal') and self.index.ntotal == 0)):
                 return "System error: Index is not built or empty, but page metadata exists. Please rebuild the index."
             return "System error: Index not initialized. Corpus might be empty or not processed."

        if self.index.ntotal == 0 :
            if len(self.page_infos_list) > 0:
                logging.warning("Index is empty, but page_infos exist. This indicates an issue with index building or all pages failed embedding.")
                return "System error: Index is empty, but document pages were found. Retrieval cannot proceed for this query."
            else:
                logging.info("No documents were found or indexed. Cannot process query.")
                return "No documents are available in the system to answer the query."

        retrieved = self.retrieve_pages(query)
        return self.answer_query_with_openai(query, retrieved)

# --- Modified run_rag_evaluation function ---
def run_rag_evaluation(
    rag_system: SimplifiedM3DOCRAG, 
    eval_data_path: str, 
    corpus_root_path_for_eval: str,
    num_concurrent_openai_calls: int = 5 
):
    try:
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            evaluation_set = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load or parse evaluation data from {eval_data_path}: {e}", exc_info=True)
        return

    logging.info(f"Loaded {len(evaluation_set)} items for RAG evaluation from {eval_data_path}.")
    
    try:
        # Use the instance's default for force_rebuild, can be overridden if necessary
        rag_system.build_or_load_index( 
            current_force_rebuild=rag_system._force_rebuild_index_default, 
            corpus_root_path=corpus_root_path_for_eval 
        )
    except ValueError as ve: # Specific error if corpus_root_path is missing and needed
        logging.error(f"Failed to initialize index for evaluation: {ve}. Make sure corpus_root_path_for_eval is correct and index can be built/loaded.")
        return
    except Exception as e_build:
        logging.error(f"Unexpected error during index build/load for evaluation: {e_build}", exc_info=True)
        return
    
    if rag_system.index is None:
        logging.error("RAG index is None after build/load attempt. Cannot proceed with evaluation.")
        return
    if rag_system.index.ntotal == 0:
        if len(rag_system.page_infos_list) == 0:
            logging.warning("RAG index and page infos are empty (no documents indexed). Evaluation will run, but expect no retrieval hits and no answers from OpenAI.")
        else: # Index empty but page_infos exist
            logging.error("RAG index is empty, but page_infos exist. This indicates a critical issue with index construction (e.g., all embeddings failed). QA stage might be unreliable.")
    
    total_queries_for_retrieval = 0
    total_retrieval_hits = 0
    detailed_e2e_results: List[Dict[str, Any]] = [] # Ensure it's typed
    
    items_to_process_for_qa: List[Dict[str, Any]] = []

    for i, eval_item in enumerate(tqdm(evaluation_set, desc="Stage 1: Retrieving pages for eval items")):
        query = eval_item.get("question")
        gt_doc_id = eval_item.get("file_name") # This is the document subfolder name
        gt_reference_pages_0indexed_str = eval_item.get("reference_page", []) # Pages are often 0-indexed strings or numbers
        gt_answer_for_reference = eval_item.get("answer", "")
        uid = eval_item.get("uid", f"item_{i}")

        if not query or not gt_doc_id:
            logging.warning(f"Skipping item UID {uid}: missing query or file_name (doc_id). Query: '{query}', Doc ID: '{gt_doc_id}'")
            detailed_e2e_results.append({
                "uid": uid, "question": query, 
                "error": "Missing query or ground truth document ID.", 
                "is_retrieval_hit": False, 
                "generated_answer": "N/A - Skipped due to missing critical input data"
            })
            continue
        
        total_queries_for_retrieval += 1
        expected_doc_id = gt_doc_id # Already the folder name

        # Convert reference pages to integers for set comparison
        try:
            gt_reference_pages_0indexed_int = [int(p) for p in gt_reference_pages_0indexed_str]
        except ValueError:
            logging.warning(f"Could not convert reference pages to int for UID {uid}: {gt_reference_pages_0indexed_str}. Assuming no specific pages for hit calculation.")
            gt_reference_pages_0indexed_int = []


        retrieved_pages_with_images: List[RetrievedPage] = rag_system.retrieve_pages(query)
        
        is_retrieval_hit = False
        retrieved_details_for_log = []
        # Ground truth set: (doc_id_folder_name, page_num_int)
        ground_truth_tuples = set((expected_doc_id, p_num) for p_num in gt_reference_pages_0indexed_int)

        for rp_with_img in retrieved_pages_with_images:
            retrieved_details_for_log.append({
                "doc_id": rp_with_img.page.doc_id, 
                "page": rp_with_img.page.page_num, # This is 0-indexed from file name
                "score": rp_with_img.score,
                "image_path": rp_with_img.page.page_image_path 
            })
            if (rp_with_img.page.doc_id, rp_with_img.page.page_num) in ground_truth_tuples:
                is_retrieval_hit = True
        
        if is_retrieval_hit:
            total_retrieval_hits += 1
            
        # Store item for batch QA processing
        items_to_process_for_qa.append({
            "uid": uid,
            "query": query,
            "retrieved_pages_for_qa": retrieved_pages_with_images, # Pass the actual retrieved pages with images
            "gt_answer_for_reference": gt_answer_for_reference,
            "expected_doc_id": expected_doc_id,
            "gt_reference_pages_0indexed": gt_reference_pages_0indexed_int, # Store int list
            "retrieved_details_for_log": retrieved_details_for_log, # For logging
            "is_retrieval_hit": is_retrieval_hit,
            "k_retrieved": len(retrieved_pages_with_images)
        })

    # Stage 2: Concurrent Answer Generation for collected items
    if items_to_process_for_qa:
        queries_and_pages_payload_for_api = [
            (item["query"], item["retrieved_pages_for_qa"]) for item in items_to_process_for_qa
        ]
        logging.info(f"Sending {len(queries_and_pages_payload_for_api)} queries for concurrent answer generation using {num_concurrent_openai_calls} workers.")
        
        generated_answers_batch = rag_system.answer_queries_concurrently(
            queries_and_pages_payload_for_api, 
            num_workers=num_concurrent_openai_calls
        )
        
        # Combine QA results with other evaluation data
        for idx, item_data_for_qa in enumerate(items_to_process_for_qa):
            generated_answer = generated_answers_batch[idx]
            detailed_e2e_results.append({
                "uid": item_data_for_qa["uid"], 
                "question": item_data_for_qa["query"],
                "ground_truth_answer_reference": item_data_for_qa["gt_answer_for_reference"],
                "ground_truth_doc_id": item_data_for_qa["expected_doc_id"], 
                "ground_truth_pages": item_data_for_qa["gt_reference_pages_0indexed"],
                "retrieved_top_k_pages": item_data_for_qa["retrieved_details_for_log"], 
                "is_retrieval_hit": item_data_for_qa["is_retrieval_hit"],
                "k_retrieved": item_data_for_qa["k_retrieved"],
                "generated_answer": generated_answer
            })
    else:
        logging.info("No valid items were prepared for QA stage (Stage 2).")

    # Calculate final metrics
    retrieval_hit_rate = (total_retrieval_hits / total_queries_for_retrieval) * 100 if total_queries_for_retrieval > 0 else 0
    
    # Sort results by UID if they were potentially added out of order (though current list append maintains order of processing)
    # detailed_e2e_results.sort(key=lambda x: x.get("uid", ""))


    # Save results
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_log_path = os.path.join(rag_system.index_storage_path, f"rag_evaluation_summary_{time_str}.txt")
    details_log_path = os.path.join(rag_system.index_storage_path, f"rag_evaluation_details_{time_str}.json")

    summary_text = (
        f"\n--- RAG Evaluation Summary ---\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Corpus Path Used for Indexing: {corpus_root_path_for_eval}\n" 
        f"Evaluation Data File: {eval_data_path}\n" 
        f"Total Queries in Eval File: {len(evaluation_set)}\n"
        f"Total Queries Attempted for Retrieval (Stage 1): {total_queries_for_retrieval}\n"
        f"Total Queries Sent for QA (Stage 2): {len(items_to_process_for_qa)}\n"
        f"Total Retrieval Hits (at least one relevant page in top K): {total_retrieval_hits}\n"
        f"Retrieval Hit Rate @K (K={rag_system.max_pages_to_retrieve}): {retrieval_hit_rate:.2f}%\n"
        f"Number of Concurrent OpenAI Workers Used: {num_concurrent_openai_calls}\n"
        f"Detailed RAG results (including generated answers) saved to: {details_log_path}\n"
        f"This summary saved to: {summary_log_path}\n"
    )
    logging.info(summary_text)
    try:
        with open(summary_log_path, 'w', encoding='utf-8') as f_sum: f_sum.write(summary_text)
    except Exception as e: logging.error(f"Failed to save evaluation summary to {summary_log_path}: {e}", exc_info=True)
    
    try:
        with open(details_log_path, 'w', encoding='utf-8') as f_res: json.dump(detailed_e2e_results, f_res, indent=4, ensure_ascii=False)
    except Exception as e: logging.error(f"Failed to save detailed RAG evaluation results to {details_log_path}: {e}", exc_info=True)


# --- Modified run_rag_evaluation function for Multi-Hop ---
def run_rag_evaluation_multi(
    rag_system: SimplifiedM3DOCRAG,
    eval_data_path: str,
    corpus_root_path_for_eval: str,
    num_concurrent_openai_calls: int = 5
):
    try:
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            evaluation_set = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load or parse evaluation data from {eval_data_path}: {e}", exc_info=True)
        return

    logging.info(f"Loaded {len(evaluation_set)} items for RAG evaluation from {eval_data_path}.")

    try:
        rag_system.build_or_load_index(
            current_force_rebuild=rag_system._force_rebuild_index_default,
            corpus_root_path=corpus_root_path_for_eval
        )
    except ValueError as ve:
        logging.error(f"Failed to initialize index for evaluation: {ve}. Make sure corpus_root_path_for_eval is correct and index can be built/loaded.")
        return
    except Exception as e_build:
        logging.error(f"Unexpected error during index build/load for evaluation: {e_build}", exc_info=True)
        return

    if rag_system.index is None:
        logging.error("RAG index is None after build/load attempt. Cannot proceed with evaluation.")
        return
    if rag_system.index.ntotal == 0:
        if len(rag_system.page_infos_list) == 0:
            logging.warning("RAG index and page infos are empty (no documents indexed). Evaluation will run, but expect no retrieval hits and no answers from OpenAI.")
        else:
            logging.error("RAG index is empty, but page_infos exist. This indicates a critical issue with index construction. QA stage might be unreliable.")

    total_queries_for_retrieval = 0
    total_multihop_retrieval_hits = 0 # Specific to new multi-hop criteria
    detailed_e2e_results: List[Dict[str, Any]] = []
    items_to_process_for_qa: List[Dict[str, Any]] = []

    for i, eval_item in enumerate(tqdm(evaluation_set, desc="Stage 1: Retrieving pages & calculating multi-hop hit")):
        main_query = eval_item.get("question")
        gt_doc_id = eval_item.get("file_name") # Document subfolder name
        gt_top_level_ref_pages_str = eval_item.get("reference_page", []) # Overall reference pages
        gt_answer_for_reference = eval_item.get("answer", "")
        uid = eval_item.get("uid", f"item_{i}")
        hop_steps = eval_item.get("steps") # List of hop dicts

        if not main_query or not gt_doc_id:
            logging.warning(f"Skipping item UID {uid}: missing main_query or file_name (doc_id). Query: '{main_query}', Doc ID: '{gt_doc_id}'")
            detailed_e2e_results.append({
                "uid": uid, "question": main_query,
                "error": "Missing main_query or ground truth document ID.",
                "is_retrieval_hit_multihop": False, # Use a distinct key for multi-hop hit
                "generated_answer": "N/A - Skipped due to missing critical input data"
            })
            continue

        total_queries_for_retrieval += 1
        expected_doc_id = gt_doc_id # This is the folder name, used for matching

        # Convert top-level reference pages for logging purposes
        try:
            gt_top_level_ref_pages_int = [int(p) for p in gt_top_level_ref_pages_str]
        except ValueError:
            logging.warning(f"UID {uid}: Could not convert top-level reference pages {gt_top_level_ref_pages_str} to int for logging.")
            gt_top_level_ref_pages_int = []

        retrieved_pages_with_images: List[RetrievedPage] = rag_system.retrieve_pages(main_query)
        retrieved_page_tuples = set()
        if retrieved_pages_with_images:
            retrieved_page_tuples = set((rp.page.doc_id, rp.page.page_num) for rp in retrieved_pages_with_images)

        # --- Multi-hop hit calculation ---
        is_multihop_hit = False # Default to False for this item
        if hop_steps and isinstance(hop_steps, list) and len(hop_steps) > 0:
            all_individual_hops_found_evidence = True # Assume true until a hop fails
            for hop_idx, hop_step_data in enumerate(hop_steps):
                hop_ref_pages_str = hop_step_data.get("reference_page", [])
                if not hop_ref_pages_str: # If a hop has no reference pages, it cannot contribute to a "hit" for that hop
                    all_individual_hops_found_evidence = False
                    logging.debug(f"UID {uid}, Hop {hop_idx}: No reference pages defined. Multi-hop criteria for this hop not met.")
                    break # This hop cannot be hit, so the multi-hop item fails

                try:
                    hop_ref_pages_int = [int(p) for p in hop_ref_pages_str]
                except ValueError:
                    logging.warning(f"UID {uid}, Hop {hop_idx}: Could not convert hop reference pages {hop_ref_pages_str} to int. Hop cannot be hit.")
                    all_individual_hops_found_evidence = False
                    break # Invalid ref pages for this hop

                # Ground truth for this specific hop: (doc_id_folder_name, page_num_int)
                hop_ground_truth_tuples = set((expected_doc_id, p_num) for p_num in hop_ref_pages_int)

                if not retrieved_page_tuples.intersection(hop_ground_truth_tuples):
                    all_individual_hops_found_evidence = False
                    logging.debug(f"UID {uid}, Hop {hop_idx}: No intersection with retrieved pages. Hop failed. Retrieved: {len(retrieved_page_tuples)}, Hop GT: {len(hop_ground_truth_tuples)}")
                    break # This hop failed, so the overall multi-hop query fails this criteria
            is_multihop_hit = all_individual_hops_found_evidence
        else:
            logging.debug(f"UID {uid}: No 'steps' found or steps are empty. Not considered a multi-hop hit by the new criteria.")
            is_multihop_hit = False # Not a multi-hop item per "steps" definition or steps are empty

        if is_multihop_hit:
            total_multihop_retrieval_hits += 1
        # --- End of Multi-hop hit calculation ---

        retrieved_details_for_log = []
        for rp_with_img in retrieved_pages_with_images:
            retrieved_details_for_log.append({
                "doc_id": rp_with_img.page.doc_id,
                "page": rp_with_img.page.page_num,
                "score": rp_with_img.score,
                "image_path": rp_with_img.page.page_image_path
            })

        items_to_process_for_qa.append({
            "uid": uid,
            "query": main_query, # Main multi-hop question for QA
            "retrieved_pages_for_qa": retrieved_pages_with_images,
            "gt_answer_for_reference": gt_answer_for_reference,
            "expected_doc_id": expected_doc_id,
            "gt_top_level_reference_pages": gt_top_level_ref_pages_int, # Log top-level for info
            "hop_steps_info": hop_steps, # Log the hop structure
            "retrieved_details_for_log": retrieved_details_for_log,
            "is_retrieval_hit_multihop": is_multihop_hit, # Store the specific multi-hop hit status
            "k_retrieved": len(retrieved_pages_with_images)
        })

    # Stage 2: Concurrent Answer Generation
    if items_to_process_for_qa:
        queries_and_pages_payload_for_api = [
            (item["query"], item["retrieved_pages_for_qa"]) for item in items_to_process_for_qa
        ]
        logging.info(f"Sending {len(queries_and_pages_payload_for_api)} queries for concurrent answer generation using {num_concurrent_openai_calls} workers.")

        generated_answers_batch = rag_system.answer_queries_concurrently(
            queries_and_pages_payload_for_api,
            num_workers=num_concurrent_openai_calls
        )

        for idx, item_data_for_qa in enumerate(items_to_process_for_qa):
            generated_answer = generated_answers_batch[idx]
            detailed_e2e_results.append({
                "uid": item_data_for_qa["uid"],
                "question": item_data_for_qa["query"],
                "ground_truth_answer_reference": item_data_for_qa["gt_answer_for_reference"],
                "ground_truth_doc_id": item_data_for_qa["expected_doc_id"],
                "ground_truth_pages_overall": item_data_for_qa["gt_top_level_reference_pages"], # Changed key for clarity
                # "hop_steps_info": item_data_for_qa["hop_steps_info"], # Optionally log all hop details
                "retrieved_top_k_pages": item_data_for_qa["retrieved_details_for_log"],
                "is_retrieval_hit_multihop": item_data_for_qa["is_retrieval_hit_multihop"],
                "k_retrieved": item_data_for_qa["k_retrieved"],
                "generated_answer": generated_answer
            })
    else:
        logging.info("No valid items were prepared for QA stage (Stage 2).")

    multihop_hit_rate = (total_multihop_retrieval_hits / total_queries_for_retrieval) * 100 if total_queries_for_retrieval > 0 else 0

    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_log_path = os.path.join(rag_system.index_storage_path, f"rag_multihop_eval_summary_{time_str}.txt")
    details_log_path = os.path.join(rag_system.index_storage_path, f"rag_multihop_eval_details_{time_str}.json")

    summary_text = (
        f"\n--- Multi-Hop RAG Evaluation Summary ---\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Corpus Path Used for Indexing: {corpus_root_path_for_eval}\n"
        f"Evaluation Data File: {eval_data_path}\n"
        f"Total Queries in Eval File: {len(evaluation_set)}\n"
        f"Total Queries Attempted for Retrieval (Stage 1): {total_queries_for_retrieval}\n"
        f"Total Queries Sent for QA (Stage 2): {len(items_to_process_for_qa)}\n"
        f"Total Multi-Hop Retrieval Hits (evidence in EVERY hop): {total_multihop_retrieval_hits}\n"
        f"Multi-Hop Retrieval Hit Rate @K (K={rag_system.max_pages_to_retrieve}): {multihop_hit_rate:.2f}%\n"
        f"Number of Concurrent OpenAI Workers Used: {num_concurrent_openai_calls}\n"
        f"Detailed RAG results saved to: {details_log_path}\n"
        f"This summary saved to: {summary_log_path}\n"
    )
    logging.info(summary_text)
    try:
        with open(summary_log_path, 'w', encoding='utf-8') as f_sum: f_sum.write(summary_text)
    except Exception as e: logging.error(f"Failed to save evaluation summary to {summary_log_path}: {e}", exc_info=True)

    try:
        with open(details_log_path, 'w', encoding='utf-8') as f_res: json.dump(detailed_e2e_results, f_res, indent=4, ensure_ascii=False)
    except Exception as e: logging.error(f"Failed to save detailed RAG evaluation results to {details_log_path}: {e}", exc_info=True)


# Example Usage:
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set, either directly or via environment variable
    OPENAI_API_KEY_TO_USE = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY_TO_USE:
        # Try the hardcoded global one if env var is not set (though os.environ should have been set at top)
        OPENAI_API_KEY_TO_USE = OPENAI_API_KEY 
    
    if not OPENAI_API_KEY_TO_USE:
        print("CRITICAL: OPENAI_API_KEY is not set. Please set it as an environment variable or in the script.")
        exit(1)
    else:
        # --- Configuration for Evaluation ---
        # Path to your root corpus folder (e.g., DOC500, which contains document subfolders like "doc1_id", "doc2_id")
        corpus_root = r'path/to/your/corpus' 
        # Path to your evaluation JSON file
        evaluation_json_file = r'path/to/your/benchmark.json'
        # Force rebuild index (True for first run or if corpus changes significantly)
        force_rebuild = False 
        # Number of concurrent OpenAI calls
        num_openai_workers = 5
        # K for retrieval (max pages to retrieve)
        k_retrieve = 5
        # ------------------------------------

        print(f"Using Image Corpus Root for Evaluation: {corpus_root}") 
        print(f"Using Evaluation Data JSON: {evaluation_json_file}")
        print(f"Force Rebuild Index: {force_rebuild}")
        print(f"Number of Concurrent OpenAI Workers: {num_openai_workers}")
        print(f"Max Pages to Retrieve (K): {k_retrieve}")


        if not os.path.isdir(corpus_root): 
            raise FileNotFoundError(f"Corpus root folder not found or is not a directory: {corpus_root}. Please check the path.")
        if not os.path.exists(evaluation_json_file):
            raise FileNotFoundError(f"Evaluation JSON file not found: {evaluation_json_file}. Please check the path.")

        try:
            rag_doc_system = SimplifiedM3DOCRAG(
                openai_api_key=OPENAI_API_KEY_TO_USE, 
                max_pages_to_retrieve=k_retrieve,
                force_rebuild_index=force_rebuild # This is the default, actual rebuild decision is in build_or_load_index
            )
            
            # The force_rebuild_index in __init__ sets the default. 
            # build_or_load_index will use that default if its current_force_rebuild param isn't changed by logic.
            # Here, we are relying on the default set during SimplifiedM3DOCRAG instantiation.

            run_rag_evaluation_multi(
                rag_system=rag_doc_system, 
                eval_data_path=evaluation_json_file, 
                corpus_root_path_for_eval=corpus_root,
                num_concurrent_openai_calls=num_openai_workers 
            )


        except Exception as main_e:
            logging.error(f"An error occurred in the main execution block: {main_e}", exc_info=True)