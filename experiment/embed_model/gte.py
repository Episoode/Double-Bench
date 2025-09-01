import json
import os
import torch
import numpy as np
import faiss
import pickle
from glob import glob
from tqdm import tqdm
from typing import List, Tuple, Dict
import argparse
from sentence_transformers import SentenceTransformer
import multiprocessing
from functools import partial

# Set multiprocessing start method for cross-platform compatibility
def set_multiprocessing_start_method():
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        # Already set, can be safely ignored
        pass

class DocumentPageEmbeddingSystem:
    def __init__(
            self,
            model_name: str = "/path/to/your/gte_Qwen2-7B-instruct",#Alibaba-NLP/gte-Qwen2-7B-instruct
            device: str = "cuda",
            batch_size: int = 8,
            max_seq_length: int = 8192
    ):
        """Initialize document page embedding system"""
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        print(f"Loading model {model_name} ...")
        # Load model in main process only
        if multiprocessing.current_process().name == 'MainProcess':
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device
            )
            self.model.max_seq_length = max_seq_length
        else:
            self.model = None  # Subprocesses will load their own model

        self.path_to_id: Dict[str, int] = {}
        self.id_to_path: Dict[int, str] = {}
        self.embeddings: List[np.ndarray] = []
        self.index = None  # FAISS index

        # Multi-GPU support
        self.multi_gpu_mode = False
        self.devices = [device]
        print("Model loaded")

    def read_folder_paths(self, txt_path: str) -> List[str]:
        """Read folder paths from txt file"""
        with open(txt_path, 'r') as f:
            folder_paths = [line.strip() for line in f if line.strip()]
        return folder_paths

    def enable_multi_gpu(self, devices: List[str]):
        """Enable multi-GPU processing mode"""
        if len(devices) > 1 and all('cuda' in d for d in devices):
            self.multi_gpu_mode = True
            self.devices = devices
            print(f"Multi-GPU mode enabled. Using devices: {', '.join(self.devices)}")
        else:
            print(f"Less than 2 GPU devices detected. Using single GPU mode ({self.device})")

    @staticmethod
    def _process_document_batch(doc_paths: List[str], model_name: str, max_seq_length: int, device: str, temp_dir: str):
        """Batch process a set of documents on a single GPU (static method for multiprocessing)"""
        local_model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        local_model.max_seq_length = max_seq_length

        all_page_texts, all_page_index_paths = [], []

        for doc_path in tqdm(doc_paths, desc=f"Aggregating text ({device})", position=int(device.split(':')[-1])):
            ocr_path = doc_path.replace('docs', 'ocr', 1)
            main_text_files = sorted(glob(os.path.join(ocr_path, 'text', '*.txt')),
                                     key=lambda x: int(os.path.basename(x).split('.')[0]))
            for page_file in main_text_files:
                try:
                    page_number = os.path.basename(page_file).split('.')[0]
                    with open(page_file, 'r', encoding='utf-8', errors='ignore') as f:
                        main_text = f.read().strip()
                    table_text = ""
                    for t_file in glob(os.path.join(ocr_path, 'table_text', f'{page_number}_*.txt')):
                        with open(t_file, 'r', encoding='utf-8', errors='ignore') as f:
                            table_text += f.read().strip() + "\n"
                    figure_text = ""
                    for f_file in glob(os.path.join(ocr_path, 'figure_text', f'{page_number}_*.txt')):
                        with open(f_file, 'r', encoding='utf-8', errors='ignore') as f:
                            figure_text += f.read().strip() + "\n"

                    combined_text = f"{main_text}\n\n{table_text}\n\n{figure_text}".strip() or " "
                    text_with_prompt = f"Represent this document for retrieval: {combined_text}"
                    page_index_path = f"{ocr_path}/{page_number}"
                    all_page_texts.append(text_with_prompt)
                    all_page_index_paths.append(page_index_path)
                except Exception as e:
                    print(f"Error processing page {ocr_path}/{page_number}: {e}")

        if not all_page_texts:
            return 0

        print(f"\nDevice {device}: Start batch encoding {len(all_page_texts)} pages ...")
        embeddings = local_model.encode(all_page_texts, batch_size=32, show_progress_bar=True)

        device_name = device.replace(':', '_')
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, f"{device_name}_paths.pkl"), 'wb') as f:
            pickle.dump(all_page_index_paths, f)
        np.save(os.path.join(temp_dir, f"{device_name}_embeddings.npy"), np.array(embeddings, dtype=np.float32))
        return len(all_page_texts)

    def _merge_results(self, temp_dir: str):
        """Merge results from multiple GPUs"""
        print("Merging results from GPUs...")
        self.path_to_id, self.id_to_path, all_embeddings_list = {}, {}, []
        offset = 0

        device_files = sorted([f for f in os.listdir(temp_dir) if f.endswith("_embeddings.npy")])
        for f_name in device_files:
            device_name = f_name.replace("_embeddings.npy", "")
            paths = pickle.load(open(os.path.join(temp_dir, f"{device_name}_paths.pkl"), 'rb'))
            embeddings = np.load(os.path.join(temp_dir, f_name))

            for i, path in enumerate(paths):
                new_id = offset + i
                self.path_to_id[path] = new_id
                self.id_to_path[new_id] = path

            all_embeddings_list.append(embeddings)
            offset += len(paths)

        if all_embeddings_list:
            self.embeddings = np.vstack(all_embeddings_list)
        print(f"Merged. Total {len(self.path_to_id)} pages.")

    def process_document_pages(self, doc_paths: List[str]) -> None:
        """Process document pages and generate embeddings (multi-GPU supported)"""
        print(f"Start processing {len(doc_paths)} documents ...")

        if self.multi_gpu_mode:
            temp_dir = os.path.join(os.getcwd(), "temp_embeddings_gte")
            os.makedirs(temp_dir, exist_ok=True)

            num_gpus = len(self.devices)
            docs_per_gpu = np.array_split(doc_paths, num_gpus)

            process_func = partial(self._process_document_batch, model_name=self.model_name,
                                   max_seq_length=self.max_seq_length, temp_dir=temp_dir)

            with multiprocessing.Pool(processes=num_gpus) as pool:
                results = pool.starmap(process_func, zip(docs_per_gpu, self.devices))

            print(f"Multi-GPU processing completed. Total {sum(results)} pages processed")
            self._merge_results(temp_dir)

        else:  # Single GPU mode
            all_page_texts, all_page_index_paths = [], []
            for doc_path in tqdm(doc_paths, desc="Aggregating all page texts"):
                ocr_path = doc_path.replace('docs', 'ocr', 1)
                main_text_files = sorted(glob(os.path.join(ocr_path, 'text', '*.txt')),
                                         key=lambda x: int(os.path.basename(x).split('.')[0]))
                for page_file in main_text_files:
                    try:
                        page_number = os.path.basename(page_file).split('.')[0]
                        with open(page_file, 'r', encoding='utf-8', errors='ignore') as f:
                            main_text = f.read().strip()
                        table_text = ""
                        for t_file in glob(os.path.join(ocr_path, 'table_text', f'{page_number}_*.txt')):
                            with open(t_file, 'r', encoding='utf-8', errors='ignore') as f:
                                table_text += f.read().strip() + "\n"
                        figure_text = ""
                        for f_file in glob(os.path.join(ocr_path, 'figure_text', f'{page_number}_*.txt')):
                            with open(f_file, 'r', encoding='utf-8', errors='ignore') as f:
                                figure_text += f.read().strip() + "\n"

                        combined_text = f"{main_text}\n\n{table_text}\n\n{figure_text}".strip() or " "
                        text_with_prompt = f"Represent this document for retrieval: {combined_text}"
                        page_index_path = f"{ocr_path}/{page_number}"
                        all_page_texts.append(text_with_prompt)
                        all_page_index_paths.append(page_index_path)
                    except Exception as e:
                        print(f"Error processing page {ocr_path}/{page_number}: {e}")

            if all_page_texts:
                print(f"Aggregated {len(all_page_texts)} pages, start batch encoding ...")
                embeddings = self.model.encode(all_page_texts, batch_size=self.batch_size, show_progress_bar=True)
                self.embeddings = np.array(embeddings, dtype=np.float32)
                for i, path in enumerate(all_page_index_paths):
                    self.path_to_id[path] = i
                    self.id_to_path[i] = path

    def save_to_disk(self, output_dir: str) -> None:
        """Save embeddings, mappings, and FAISS index to disk"""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'path_to_id.pkl'), 'wb') as f: pickle.dump(self.path_to_id, f)
        with open(os.path.join(output_dir, 'id_to_path.pkl'), 'wb') as f: pickle.dump(self.id_to_path, f)

        if len(self.embeddings) > 0:
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            vector_dim = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(vector_dim)
            faiss.normalize_L2(embeddings_array)
            index.add(embeddings_array)
            faiss.write_index(index, os.path.join(output_dir, 'vector.index'))
            print(f"FAISS index created and saved. Vector dimension: {vector_dim}")

        print(f"All data saved to: {output_dir}")
        print(f"Total indexed document pages: {len(self.path_to_id)}")

    def load_from_disk(self, input_dir: str) -> None:
        """Load mappings and FAISS index from disk"""
        print(f"Loading data from {input_dir} ...")
        with open(os.path.join(input_dir, 'path_to_id.pkl'), 'rb') as f:
            self.path_to_id = pickle.load(f)
        with open(os.path.join(input_dir, 'id_to_path.pkl'), 'rb') as f:
            self.id_to_path = pickle.load(f)
        index_path = os.path.join(input_dir, 'vector.index')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Successfully loaded FAISS index with {self.index.ntotal} vectors.")
        else:
            raise FileNotFoundError(f"Error: vector.index not found in {input_dir}.")

    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search efficiently in loaded FAISS index by text"""
        if self.index is None: raise ValueError("Index not loaded. Call load_from_disk first.")
        if not query_text or not query_text.strip(): return []
        try:
            # GTE model search with prompt_name='query'
            query_embedding = self.model.encode([query_text], prompt_name="query")
            query_norm = np.array(query_embedding, dtype=np.float32)
            faiss.normalize_L2(query_norm)

            actual_top_k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_norm, actual_top_k)

            results = [(self.id_to_path[idx], float(sim)) for idx, sim in zip(indices[0], similarities[0]) if idx != -1]
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

# --- Wrapper Functions ---

def embed_and_index_documents(txt_path: str, output_dir: str, devices: List[str]):
    system = DocumentPageEmbeddingSystem(device=devices[0])
    if len(devices) > 1:
        system.enable_multi_gpu(devices)
    doc_paths = system.read_folder_paths(txt_path)
    system.process_document_pages(doc_paths)
    system.save_to_disk(output_dir)

def search_documents(model_dir: str, query_text: str, top_k: int, device: str):
    system = DocumentPageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)
    results = system.search_by_text(query_text, top_k=top_k)
    print(f"\nQuery: '{query_text}'")
    print(f"Top {len(results)} results found:")
    for i, (path, score) in enumerate(results):
        print(f"  {i + 1}. Path: {path} (Similarity: {score:.4f})")

def process_json_file(json_file_path: str, output_json_path: str, model_dir: str, top_k: int, device: str):
    system = DocumentPageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Processing {len(data)} JSON entries...")
    for item in tqdm(data, desc="Processing JSON"):
        question = item.get("question") or item.get("final_question")
        if question:
            results = system.search_by_text(question, top_k=top_k)
            item["retrieval_pages"] = [path for path, _ in results]
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Processing finished. Results saved to {output_json_path}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='GTE-Qwen2 Document Page Embedding and Retrieval System (Multi-GPU)')
    parser.add_argument('--mode', type=str, choices=['embed', 'search', 'process_json'], required=True,
                        help='Mode: embed (embedding and indexing), search (interactive search), process_json (batch process JSON)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Main device or device used for search (e.g., cuda:0)')
    parser.add_argument('--devices', type=str, nargs='*',
                        help='[embed mode] List of GPU devices to use for embedding (e.g., cuda:0 cuda:1 cuda:2)')

    parser.add_argument('--output_dir', type=str,
                        help='[embed mode] Output directory for index, or [search/process_json mode] directory for loading index')
    parser.add_argument('--txt_path', type=str, help='[embed mode] txt file containing document paths')
    parser.add_argument('--query', type=str, help='[search mode] Query text')
    parser.add_argument('--json_file', type=str, help='[process_json mode] Input JSON file')
    parser.add_argument('--output_json', type=str, help='[process_json mode] Output JSON file')
    parser.add_argument('--top_k', type=int, default=10, help='Number of most similar results to return')
    args = parser.parse_args()

    if args.mode == 'embed':
        if not all([args.txt_path, args.output_dir]):
            parser.error("Embed mode (--mode embed) requires --txt_path and --output_dir arguments")
        devices = args.devices if args.devices else [args.device]
        embed_and_index_documents(args.txt_path, args.output_dir, devices)

    elif args.mode == 'search':
        if not all([args.output_dir, args.query]):
            parser.error("Search mode (--mode search) requires --output_dir and --query arguments")
        search_documents(args.output_dir, args.query, args.top_k, args.device)

    elif args.mode == 'process_json':
        if not all([args.output_dir, args.json_file, args.output_json]):
            parser.error("Process_json mode (--mode process_json) requires --output_dir, --json_file, and --output_json arguments")
        process_json_file(args.json_file, args.output_json, args.output_dir, args.top_k, args.device)

if __name__ == "__main__":
    set_multiprocessing_start_method()
    main()