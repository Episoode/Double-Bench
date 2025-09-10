import os
import glob
import re
from PIL import Image  # For loading images
import numpy as np
import pickle  # For checkpointing
import time  # For simple progress/timing
from gme_inference import GmeQwen2VL  
from FlagEmbedding import BGEM3FlagModel
import torch
import json


# --- MODIFIED Helper Function to Parse Filenames ---
def parse_document_filename(filepath):
    """
    Parses a filepath like '/.../docs/English/0653/005.jpg'
    to extract the document ID and page number.

    Returns:
        tuple: (doc_id, page_num) e.g., ('0653', 5)
    """
    try:
        # Extract page number from filename, e.g., '005.jpg' -> 5
        basename = os.path.basename(filepath)
        name_part, _ = os.path.splitext(basename)
        page_num = int(name_part)

        # The doc_id is the name of the parent directory, e.g., '/.../0653' -> '0653'
        doc_id = os.path.basename(os.path.dirname(filepath))
        return doc_id, page_num
    except (ValueError, IndexError) as e:
        print(f"WARN: Could not parse document ID and page from '{filepath}'. Error: {e}. Defaulting name and page 0.")
        name_part, _ = os.path.splitext(os.path.basename(filepath))
        return name_part, 0


class DocumentInformationRetriever:
    def __init__(self, corpus_base_path, query_metadata_path, checkpoint_path="retriever_checkpoint.pkl", checkpoint_interval=100,
                 text_batch_size=16, image_batch_size=2):
        self.corpus_base_path = corpus_base_path
        self.docs_root_path = os.path.join(self.corpus_base_path, 'merged_doc')
        self.ocr_root_path = os.path.join(self.corpus_base_path, 'ocr')
        self.query_metadata_path = query_metadata_path
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval

        self.text_batch_size = text_batch_size
        self.image_batch_size = image_batch_size

        # For storing embeddings and their corresponding metadata separately
        self._image_embeddings_list = []
        self._image_metadata_list = []
        self._text_embeddings_list = []
        self._text_metadata_list = []
        self._processed_files = set()  # To keep track of files already processed

        self.retrieval_device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.query_metadata_path):
            raise FileNotFoundError(f"Benchmark metadata JSON not found: {self.query_metadata_path}")

        # Initialize model wrappers
        try:
            self.gme_model = GmeQwen2VL(
                model_name='Alibaba-NLP/gme-Qwen2-VL-7B-Instruct',
                device=self.retrieval_device
            )
            # self.bge_model = BGEM3FlagModel(
            #     use_fp16=True,
            #     devices=self.retrieval_device
            # )
        except Exception as e:
            print(f"ERROR: Failed to load models: {e}")
            print("INFO: DocumentInformationRetriever initialization failed due to model loading issues.")
            raise

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                # Load separated embeddings and metadata
                self._image_embeddings_list = checkpoint_data.get('image_embeddings_list', [])
                self._image_metadata_list = checkpoint_data.get('image_metadata_list', [])
                self._text_embeddings_list = checkpoint_data.get('text_embeddings_list', [])
                self._text_metadata_list = checkpoint_data.get('text_metadata_list', [])
                self._processed_files = checkpoint_data.get('processed_files', set())

                # Integrity check
                if len(self._image_embeddings_list) != len(self._image_metadata_list):
                    print("WARN: Mismatch between loaded image embeddings and metadata. Resetting image cache.")
                    self._image_embeddings_list, self._image_metadata_list = [], []
                if len(self._text_embeddings_list) != len(self._text_metadata_list):
                    print("WARN: Mismatch between loaded text embeddings and metadata. Resetting text cache.")
                    self._text_embeddings_list, self._text_metadata_list = [], []

                print(f"INFO: Loaded checkpoint from {self.checkpoint_path}. "
                      f"Resuming with {len(self._image_embeddings_list)} image embeddings, "
                      f"{len(self._text_embeddings_list)} text embeddings. "
                      f"{len(self._processed_files)} files already processed.")
                return True
            except Exception as e:
                print(f"ERROR: Could not load checkpoint from {self.checkpoint_path}: {e}. Starting fresh.")
                self._image_embeddings_list, self._image_metadata_list = [], []
                self._text_embeddings_list, self._text_metadata_list = [], []
                self._processed_files = set()
                return False
        return False

    def _save_checkpoint(self):
        try:
            checkpoint_data = {
                'image_embeddings_list': self._image_embeddings_list,
                'image_metadata_list': self._image_metadata_list,
                'text_embeddings_list': self._text_embeddings_list,
                'text_metadata_list': self._text_metadata_list,
                'processed_files': self._processed_files
            }
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception as e:
            print(f"ERROR: Could not save checkpoint to {self.checkpoint_path}: {e}")

    def preprocess_corpus(self, force_reprocess=False):
        print("INFO: Starting corpus preprocessing...")
        
        # Handle force reprocess or load checkpoint
        if force_reprocess and os.path.exists(self.checkpoint_path):
            print(f"INFO: Force reprocess requested. Deleting existing checkpoint.")
            os.remove(self.checkpoint_path)
            # Reset all lists
            self.__dict__.update({
                '_image_embeddings_list': [], '_image_metadata_list': [],
                '_text_embeddings_list': [], '_text_metadata_list': [],
                '_processed_files': set()
            })
        elif not force_reprocess:
            self._load_checkpoint()

        # --- Document Discovery ---
        print(f"INFO: Scanning for all documents in {self.docs_root_path}...")
        all_doc_img_folders = [
            p for p in glob.glob(os.path.join(self.docs_root_path, '*', '*')) 
            if os.path.isdir(p)
        ]
        print(f"INFO: Found {len(all_doc_img_folders)} total document folders to process.")
        
        # Collect all image files
        all_img_files = []
        for doc_img_folder in all_doc_img_folders:
             all_img_files.extend(glob.glob(os.path.join(doc_img_folder, "*.jpg")))
             all_img_files.extend(glob.glob(os.path.join(doc_img_folder, "*.png")))
        all_img_files = sorted(list(set(all_img_files)))  # Ensure uniqueness and order
        print(f"INFO: Discovered a total of {len(all_img_files)} image files.")

        # --- Text Processing ---
        files_processed_in_session = 0
        text_batch_content, text_batch_metadata = [], []
        
        print(f"INFO: Aggregating and processing text content...")
        for doc_img_folder in all_doc_img_folders:
            doc_id = os.path.basename(doc_img_folder)
            doc_ocr_folder = doc_img_folder.replace(self.docs_root_path, self.ocr_root_path, 1)
            page_texts = {}

            # Paths for text components
            paths = {
                'text': os.path.join(doc_ocr_folder, 'text'),
                'table_text': os.path.join(doc_ocr_folder, 'table_text'),
                'figure_text': os.path.join(doc_ocr_folder, 'figure_text')
            }

            # 1. Read base text
            if os.path.isdir(paths['text']):
                for fpath in glob.glob(os.path.join(paths['text'], "*.txt")):
                    try:
                        page_num = int(os.path.splitext(os.path.basename(fpath))[0])
                        with open(fpath, 'r', encoding='utf-8') as f:
                            page_texts[page_num] = f.read()
                    except (ValueError, IOError): 
                        continue
            
            # 2. Append table and figure text
            for component_type in ['table_text', 'figure_text']:
                if os.path.isdir(paths[component_type]):
                    component_files = glob.glob(os.path.join(paths[component_type], "*.txt"))
                    try:
                        component_files.sort(key=lambda f: [
                            int(p) for p in os.path.splitext(os.path.basename(f))[0].split('_')
                        ])
                    except ValueError:
                        print(f"WARN: Could not sort files in {paths[component_type]} due to unexpected naming. Processing in default order.")

                    for fpath in component_files:
                        try:
                            page_num = int(os.path.splitext(os.path.basename(fpath))[0].split('_')[0])
                            with open(fpath, 'r', encoding='utf-8') as f:
                                page_texts.setdefault(page_num, "")
                                page_texts[page_num] += "\n" + f.read()
                        except (ValueError, IOError, IndexError) as e: 
                            print(e)
                            exit(1)
            
            # 3. Add aggregated texts to processing batch
            for page_num, aggregated_text in sorted(page_texts.items()):
                page_id_path = os.path.join(paths['text'], f"{page_num:03d}.txt")
                if page_id_path in self._processed_files or not aggregated_text.strip():
                    continue

                text_batch_content.append(aggregated_text)
                text_batch_metadata.append({
                    'path': page_id_path, 
                    'doc_name': doc_id, 
                    'page': page_num, 
                    'original_text': aggregated_text
                })

                # Process the batch when it's full
                if len(text_batch_content) >= self.text_batch_size:
                    print(f"INFO: Processing batch of {len(text_batch_content)} aggregated text pages...")
                    try:
                        if self.retrieval_device.startswith("cuda"):
                            with torch.cuda.device(self.retrieval_device):
                                torch.cuda.empty_cache()
                                embeddings = self.model.get_text_embeddings(text_batch_content, batch_size=len(text_batch_content))
                        else:
                            embeddings = self.model.get_text_embeddings(text_batch_content, batch_size=len(text_batch_content))

                        if embeddings is not None and len(embeddings) == len(text_batch_content):
                            for idx, embedding in enumerate(embeddings):
                                self._text_embeddings_list.append(embedding)
                                meta = text_batch_metadata[idx]
                                self._text_metadata_list.append({
                                    'path': meta['path'],
                                    'doc_name': meta['doc_name'],
                                    'page': meta['page'],
                                    'text': meta['original_text']
                                })
                                self._processed_files.add(meta['path'])
                                files_processed_in_session += 1
                        else:
                            print("WARN: Embedding batch failed. Skipping.")
                            for meta in text_batch_metadata:
                                self._processed_files.add(meta['path'])
                                files_processed_in_session += 1
                        
                        if files_processed_in_session >= self.checkpoint_interval:
                            self._save_checkpoint()
                            files_processed_in_session = 0
                    except Exception as e:
                        print(f"ERROR: Could not process text batch: {e}")
                        for meta in text_batch_metadata:
                            self._processed_files.add(meta['path'])
                            files_processed_in_session += 1
                    finally:
                        text_batch_content, text_batch_metadata = [], []
        
        # Process any remaining text files in the last batch
        if text_batch_content:
            print(f"INFO: Processing final batch of {len(text_batch_content)} aggregated text pages...")
            try:
                embeddings = self.bge_model.encode(
                    text_batch_content,
                    batch_size=len(text_batch_content),
                    max_length=8192
                )['dense_vecs']
                
                if embeddings is not None and len(embeddings) == len(text_batch_content):
                    for idx, embedding in enumerate(embeddings):
                        self._text_embeddings_list.append(embedding)
                        meta = text_batch_metadata[idx]
                        self._text_metadata_list.append({
                            'path': meta['path'],
                            'doc_name': meta['doc_name'],
                            'page': meta['page'],
                            'text': meta['original_text']
                        })
                        self._processed_files.add(meta['path'])
                        files_processed_in_session += 1
            except Exception as e:
                print(f"ERROR: Could not process final text batch: {e}")
            finally:
                text_batch_content, text_batch_metadata = [], []

        print(f"INFO: Text file processing phase complete. Total text embeddings: {len(self._text_embeddings_list)}")
        
        if files_processed_in_session > 0:
            self._save_checkpoint()


        # --- Step 3: Process Image Files (This part remains unchanged) ---
        # --- Process Image Files with Batching ---
        print(f"INFO: Scanning image files...")
        img_files_to_process = [f for f in all_img_files if f not in self._processed_files]

        if not img_files_to_process and self._processed_files.intersection(all_img_files) == set(all_img_files) and self._image_embeddings_list:
            print("INFO: All image files seem to be processed based on checkpoint and image embeddings exist.")
        elif not img_files_to_process and all_img_files:
            print("INFO: No new image files to process.")
        elif not all_img_files:
            print("INFO: No image files found based on the benchmark file.")

        image_batch_pil, image_batch_metadata = [], []
        for i, img_filepath in enumerate(img_files_to_process):
            doc_name, page = parse_document_filename(img_filepath)
            try:
                img = Image.open(img_filepath).convert('RGB')
                image_batch_pil.append(img)
                image_batch_metadata.append({'path': img_filepath, 'doc_name': doc_name, 'page': page})

                if len(image_batch_pil) == self.image_batch_size or (i == len(img_files_to_process) - 1 and image_batch_pil):
                    print(f"INFO: Processing batch of {len(image_batch_pil)} image files...")
                    try:
                        embeddings = self.gme_model.get_image_embeddings(image_batch_pil, batch_size=len(image_batch_pil))
                        if embeddings is not None and len(embeddings) == len(image_batch_pil):
                            for idx, embedding in enumerate(embeddings):
                                self._image_embeddings_list.append(embedding)
                                meta = image_batch_metadata[idx]
                                self._image_metadata_list.append({'path': meta['path'], 'doc_name': meta['doc_name'], 'page': meta['page']})
                                self._processed_files.add(meta['path'])
                                files_processed_in_session += 1
                        else:
                            print(f"WARN: GME embedding batch failed. Skipping.")
                            for meta in image_batch_metadata: 
                                self._processed_files.add(meta['path'])
                                files_processed_in_session += 1
                    
                        if files_processed_in_session >= self.checkpoint_interval:
                            self._save_checkpoint()
                            files_processed_in_session = 0
                    except Exception as e:
                        print(f"ERROR: Could not process image batch: {e}")
                        for meta in image_batch_metadata: 
                            self._processed_files.add(meta['path'])
                            files_processed_in_session += 1
                    finally:
                        image_batch_pil, image_batch_metadata = [], []
            except Exception as e:
                print(f"ERROR: Could not read or prepare image file {img_filepath}: {e}")
                self._processed_files.add(img_filepath)
                files_processed_in_session += 1

        print(f"INFO: Image file processing phase complete. Total image embeddings: {len(self._image_embeddings_list)}")
        if files_processed_in_session > 0:
            self._save_checkpoint()

        print(f"INFO: Corpus preprocessing complete. Final counts: {len(self._text_embeddings_list)} text embeddings, "
          f"{len(self._image_embeddings_list)} image embeddings.")

    def retrieve_top_m_images(self, query, top_m):
        if not self._image_embeddings_list:
            print("WARN: Image embeddings list is empty.")
            return []
        if top_m <= 0: return []

        query_embedding = self.gme_model.get_text_embeddings([query]) # This should return a single embedding
        if query_embedding is None:
            print("ERROR: Failed to generate GME query embedding for image retrieval.")
            return []

        query_embedding = query_embedding.reshape(1, -1)  # shape: (1, D)
        
        # Ensure embeddings are in a 2D NumPy array for dot product
        corpus_embeddings_np = np.array([emb.flatten() for emb in self._image_embeddings_list]) # shape: (N, D)
        if corpus_embeddings_np.ndim == 1: # if only one image in corpus
            corpus_embeddings_np = corpus_embeddings_np.reshape(1, -1)

        if corpus_embeddings_np.size == 0: return []

        similarities = query_embedding @ corpus_embeddings_np.T  # shape: (1, N)
        similarities = similarities.flatten()  # shape: (N,)
        
        actual_top_m = min(top_m, len(similarities))
        if actual_top_m == 0: return []

        # top_indices = np.argsort(similarities)[::-1][:actual_top_m]
        top_indices = torch.topk(torch.tensor(similarities), k=actual_top_m, largest=True).indices.tolist()

        return [{
            'image_path': self._image_metadata_list[idx]['path'],
            'doc_name': self._image_metadata_list[idx]['doc_name'],
            'page': self._image_metadata_list[idx]['page'],
            'similarity_score': float(similarities[idx])
        } for idx in top_indices]

    def retrieve_top_n_texts(self, query, top_n):
        if not self._text_embeddings_list:
            print("WARN: Text embeddings list is empty.")
            return []
        if top_n <= 0: return []

        query_embedding = self.bge_model.encode(query, batch_size=1, max_length=8192)['dense_vecs']
        if query_embedding is None:
            print("ERROR: Failed to generate BGE query embedding.")
            return []

        query_embedding = query_embedding.reshape(1, -1)
        corpus_embeddings_np = np.array([emb.flatten() for emb in self._text_embeddings_list])
        if corpus_embeddings_np.ndim == 1: 
            corpus_embeddings_np = corpus_embeddings_np.reshape(1, -1)
        if corpus_embeddings_np.size == 0: 
            return []

        similarities = (query_embedding @ corpus_embeddings_np.T).flatten()
        actual_top_n = min(top_n, len(similarities))
        if actual_top_n == 0: 
            return []

        top_indices = np.argsort(similarities)[::-1][:actual_top_n]

        return [{
            'text': self._text_metadata_list[idx]['text'],
            'doc_name': self._text_metadata_list[idx]['doc_name'],
            'page': self._text_metadata_list[idx]['page'],
            'text_path': self._text_metadata_list[idx]['path'],
            'similarity_score': float(similarities[idx])
        } for idx in top_indices]

    def search_documents(self, query, top_m_images, top_n_texts):
        print(f"\nINFO: Searching for query: '{query}'")
        image_results = self.retrieve_top_m_images(query, top_m_images)
        text_results = self.retrieve_top_n_texts(query, top_n_texts)
        return {
            "retrieved_images": image_results,
            "retrieved_texts": text_results
        }

    def _standardize_ref_pages(self, ref_pages_raw, query_for_warning=""):
        """Helper function to standardize reference page formats."""
        if isinstance(ref_pages_raw, int):
            return [ref_pages_raw]
        elif isinstance(ref_pages_raw, list) and all(isinstance(p, int) for p in ref_pages_raw):
            return ref_pages_raw
        elif isinstance(ref_pages_raw, list) and any(isinstance(p, str) for p in ref_pages_raw):
            try:
                return [int(p) for p in ref_pages_raw]
            except ValueError:
                warning_query_part = f" for query '{query_for_warning}'" if query_for_warning else ""
                print(
                    f"WARN: Could not convert all string page numbers in reference_page to int{warning_query_part}. Treating as empty.")
                return []
        else:
            return []

    def evaluation_single(self, query_metadata, top_m_images, top_n_texts, log_file_path="evaluation_log_single.json"):
        total_queries = len(query_metadata)
        if total_queries == 0: return 0.0
        hit_count, evaluation_logs = 0, []
        print(f"\nINFO: Starting single-hop evaluation for {total_queries} queries...")

        for i, item in enumerate(query_metadata):
            query = item['question']
            try:
                # Get GT doc ID from the new 'doc_path' field
                ground_doc_name = os.path.basename(item['doc_path'])
            except KeyError:
                print(f"WARN: Skipping query {i + 1} due to missing 'doc_path'.")
                continue

            ground_doc_ref_pages = self._standardize_ref_pages(item.get('reference_page', []), query)
            results = self.search_documents(query, top_m_images, top_n_texts)
            query_hit = False

            # Check image and text results
            retrieved_items = results.get("retrieved_images", []) + results.get("retrieved_texts", [])
            for retrieved in retrieved_items:
                if retrieved['doc_name'] == ground_doc_name and retrieved['page'] in ground_doc_ref_pages:
                    query_hit = True
                    break

            if query_hit: 
                hit_count += 1

            evaluation_logs.append({
                "query_id": item.get('uid', i + 1),
                "query": query,
                "ground_truth": {
                    "doc_id": ground_doc_name,
                    "reference_pages": ground_doc_ref_pages
                },
                "results": results,
                "hit_status": "HIT" if query_hit else "MISS"
            })

        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_logs, f, indent=4, ensure_ascii=False)
        hit_rate = (hit_count / total_queries) if total_queries > 0 else 0.0
        print(f"INFO: Single-hop evaluation complete. Hit Rate: {hit_rate:.4f}")
        return hit_rate

    def evaluation_multi(self, query_metadata, top_m_images, top_n_texts, log_file_path="evaluation_log_multi.json"):
        total_queries = len(query_metadata)
        if total_queries == 0: return 0.0
        hit_count, evaluation_logs = 0, []
        print(f"\nINFO: Starting multi-hop evaluation for {total_queries} queries...")

        for i, item in enumerate(query_metadata):
            main_query = item['question']
            try:
                # Get GT doc ID from the new 'doc_path' field
                original_ground_doc_name = os.path.basename(item['doc_path'])
            except KeyError:
                print(f"WARN: Skipping query {i + 1} due to missing 'doc_path'.")
                continue

            steps = item.get('steps', [])
            if not steps:
                print(f"WARN: Query '{main_query}' has no steps. Skipping.")
                continue

            hop_ground_truth_pages = [set(self._standardize_ref_pages(s.get('reference_page', []))) for s in steps]
            results = self.search_documents(main_query, top_m_images, top_n_texts)
            hops_hit_flags = [False] * len(steps)

            retrieved_items = results.get("retrieved_images", []) + results.get("retrieved_texts", [])
            for retrieved in retrieved_items:
                # Compare retrieved doc ID ('0653') with ground truth doc ID
                if retrieved.get('doc_name') == original_ground_doc_name:
                    for hop_idx, gt_pages_for_hop in enumerate(hop_ground_truth_pages):
                        if not hops_hit_flags[hop_idx] and retrieved.get('page') in gt_pages_for_hop:
                            hops_hit_flags[hop_idx] = True

            overall_query_hit = all(hops_hit_flags)
            if overall_query_hit: 
                hit_count += 1

            evaluation_logs.append({
                "query_id": item.get('uid', i + 1),
                "query": main_query,
                "ground_truth_doc_id": original_ground_doc_name,
                "steps": item.get('steps'),
                "hops_hit_status": hops_hit_flags,
                "hit_status": "HIT" if overall_query_hit else "MISS",
                "results": results
            })

        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_logs, f, indent=4, ensure_ascii=False)
        hit_rate = (hit_count / total_queries) if total_queries > 0 else 0.0
        print(f"INFO: Multi-hop evaluation complete. Hit Rate: {hit_rate:.4f}")
        return hit_rate


# --- MODIFIED Example Usage ---
if __name__ == '__main__':
    corpus_base_path = 'Benchmark'

    # Path to your new benchmark JSON file
    query_metadata_path = r'multihop.json'

    # Instantiate the retriever with the path to the benchmark JSON
    retriever = DocumentInformationRetriever(
        corpus_base_path=corpus_base_path,
        query_metadata_path=query_metadata_path,
        checkpoint_path='./checkpoints/retriever_checkpoint_new.pkl',
        checkpoint_interval=32,
        text_batch_size=8,
        image_batch_size=2
    )

    # Preprocess the corpus. The method now discovers files from the JSON.
    retriever.preprocess_corpus()

    # Load the metadata for evaluation
    try:
        with open(query_metadata_path, 'r', encoding='utf-8') as f:
            query_metadata = json.load(f)
    except Exception as e:
        print(f"CRITICAL: Could not load query metadata from {query_metadata_path} for evaluation. Exiting. Error: {e}")
        exit()

    # Call the modified evaluation function
    retriever.evaluation_multi(
        query_metadata,
        top_m_images=5,
        top_n_texts=5,
        log_file_path="evaluation_log_multi.json"
    )
