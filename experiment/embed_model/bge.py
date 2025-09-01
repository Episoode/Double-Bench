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

class DocumentPageEmbeddingSystem:
    def __init__(
            self,
            model_name: str = "/path/to/your/bge-m3",#BAAI/bge-m3
            device: str = "cuda",
            batch_size: int = 8
    ):
        """Initialize Document Page Embedding System"""
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        print(f"Loading {model_name} model using sentence-transformers ...")
        self.model = SentenceTransformer(model_name, device=device)

        self.path_to_id: Dict[str, int] = {}
        self.id_to_path: Dict[int, str] = {}
        self.embeddings: List[np.ndarray] = []
        self.index = None  # For storing loaded FAISS index
        print("Model loaded successfully.")

    def read_folder_paths(self, txt_path: str) -> List[str]:
        """Read folder paths from a txt file"""
        with open(txt_path, 'r') as f:
            folder_paths = [line.strip() for line in f if line.strip()]
        return folder_paths

    def process_document_pages(self, doc_paths: List[str]) -> None:
        """Process document pages and generate embeddings (batch optimized)"""
        print(f"Start processing {len(doc_paths)} documents...")
        all_page_texts, all_page_index_paths = [], []
        print("Step 1/3: Aggregating all page texts...")
        for doc_path in tqdm(doc_paths, desc="Aggregating pages"):
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
        print(f"\nStep 2/3: Aggregated {len(all_page_texts)} pages, start batch embedding...")
        if all_page_texts:
            all_embeddings = self.model.encode(all_page_texts, batch_size=self.batch_size, show_progress_bar=True)
            print("\nStep 3/3: Saving mappings and embeddings...")
            for i, page_path in enumerate(all_page_index_paths):
                self.path_to_id[page_path] = i
                self.id_to_path[i] = page_path
                self.embeddings.append(all_embeddings[i])

    def save_to_disk(self, output_dir: str) -> None:
        """Save embeddings and mappings to disk"""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'path_to_id.pkl'), 'wb') as f: pickle.dump(self.path_to_id, f)
        with open(os.path.join(output_dir, 'id_to_path.pkl'), 'wb') as f: pickle.dump(self.id_to_path, f)
        if self.embeddings:
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            vector_dim = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(vector_dim)
            faiss.normalize_L2(embeddings_array)
            index.add(embeddings_array)
            faiss.write_index(index, os.path.join(output_dir, 'vector.index'))
            print(f"FAISS index created and saved. Vector dimension: {vector_dim}")
        print(f"All data saved to: {output_dir}")
        print(f"Total {len(self.path_to_id)} document pages indexed.")

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
            raise FileNotFoundError(f"Error: vector.index file not found in {input_dir}.")

    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Efficient search in loaded FAISS index by text"""
        if self.index is None:
            raise ValueError("Index not loaded. Please call load_from_disk first.")
        if not query_text or not query_text.strip(): return []
        try:
            query_with_prompt = f"Represent this document for retrieval: {query_text}"
            query_embedding = self.model.encode(query_with_prompt)
            query_norm = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_norm)
            actual_top_k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_norm, actual_top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    results.append((self.id_to_path[idx], float(similarities[0][i])))
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

# --- Wrapper Functions ---

def embed_and_index_documents(txt_path: str, output_dir: str, device: str):
    system = DocumentPageEmbeddingSystem(device=device)
    doc_paths = system.read_folder_paths(txt_path)
    print(f"Read {len(doc_paths)} document paths from {txt_path}")
    system.process_document_pages(doc_paths)
    system.save_to_disk(output_dir)

def search_images(model_dir: str, query_text: str, top_k: int, device: str):
    system = DocumentPageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)
    results = system.search_by_text(query_text, top_k=top_k)
    print(f"\nQuery: '{query_text}'")
    print(f"Top {len(results)} results found:")
    for i, (path, score) in enumerate(results):
        print(f"  {i + 1}. Path: {path} (Similarity: {score:.4f})")

def process_json_file(json_file_path: str, output_json_path: str, model_dir: str, top_k: int, device: str):
    """Process JSON file, retrieve related document pages for each question and add to JSON"""
    system = DocumentPageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Start processing {len(data)} JSON entries...")
    for item in tqdm(data, desc="Processing JSON"):
        try:
            question = item.get("question") or item.get("final_question")
            if not question: continue
            results = system.search_by_text(question, top_k=top_k)
            item["retrieval_pages"] = [path for path, _ in results]
        except Exception as e:
            print(f"Error processing entry: {item}. Error: {e}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Processing completed. Results saved to {output_json_path}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='BGE M3 Document Page Embedding and Retrieval System (Sentence-Transformers version)')
    parser.add_argument('--mode', type=str, choices=['embed', 'search', 'process_json'], required=True,
                        help='Mode: embed (embedding and indexing), search (interactive search), process_json (batch process JSON)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda, cuda:0, cpu)')
    parser.add_argument('--output_dir', type=str,
                        help='[embed mode] Output directory for index, or [search/process_json mode] directory for loading index')
    # Args for 'embed'
    parser.add_argument('--txt_path', type=str, help='[embed mode] txt file containing document paths')
    # Args for 'search'
    parser.add_argument('--query', type=str, help='[search mode] Query text')
    # Args for 'process_json'
    parser.add_argument('--json_file', type=str, help='[process_json mode] Input JSON file')
    parser.add_argument('--output_json', type=str, help='[process_json mode] Output JSON file')
    # General args
    parser.add_argument('--top_k', type=int, default=10, help='Number of most similar results to return')
    args = parser.parse_args()

    if args.mode == 'embed':
        if not all([args.txt_path, args.output_dir]):
            parser.error("Embed mode (--mode embed) requires --txt_path and --output_dir arguments")
        embed_and_index_documents(args.txt_path, args.output_dir, args.device)
    elif args.mode == 'search':
        if not all([args.output_dir, args.query]):
            parser.error("Search mode (--mode search) requires --output_dir and --query arguments")
        search_images(args.output_dir, args.query, args.top_k, args.device)
    elif args.mode == 'process_json':
        if not all([args.output_dir, args.json_file, args.output_json]):
            parser.error("Process_json mode (--mode process_json) requires --output_dir, --json_file, and --output_json arguments")
        process_json_file(args.json_file, args.output_json, args.output_dir, args.top_k, args.device)

if __name__ == "__main__":
    main()