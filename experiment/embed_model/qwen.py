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


class DocumentPageEmbeddingSystemQwen3:
    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-Embedding-4B",
            device: str = "cuda",  # This device param will be overridden by device_map for multi-GPU
            batch_size: int = 4
    ):
        """Initialize document page embedding system (Qwen3 version)"""
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        print(f"Loading model from local path and distributing to multiple GPUs: {model_name} ...")

        # === Key change: multi-GPU model loading ===
        model_kwargs = {
            'attn_implementation': 'flash_attention_2',
            "device_map": "auto",
            "torch_dtype": torch.bfloat16
        }
        self.model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,  # Multi-GPU config
            trust_remote_code=True
        )
        # ===========================================

        self.path_to_id: Dict[str, int] = {}
        self.id_to_path: Dict[int, str] = {}
        self.embeddings: List[np.ndarray] = []
        self.index = None  # For FAISS index
        print("Model loaded with multi-GPU support!")

    def scan_ocr_documents(self, ocr_dir: str) -> List[str]:
        """Scan OCR directory structure: ocr/language/doc1, doc2, ..."""
        doc_paths = []
        if not os.path.exists(ocr_dir):
            print(f"OCR directory {ocr_dir} does not exist!")
            return doc_paths

        # Traverse ocr/language/doc structure
        for lang_dir in os.listdir(ocr_dir):
            lang_path = os.path.join(ocr_dir, lang_dir)
            if os.path.isdir(lang_path):
                for doc_name in os.listdir(lang_path):
                    doc_path = os.path.join(lang_path, doc_name)
                    if os.path.isdir(doc_path):
                        # Check if it has required subdirectories
                        if (os.path.exists(os.path.join(doc_path, 'text')) or
                                os.path.exists(os.path.join(doc_path, 'table_text')) or
                                os.path.exists(os.path.join(doc_path, 'figure_text'))):
                            doc_paths.append(doc_path)

        print(f"Found {len(doc_paths)} documents in OCR directory")
        return doc_paths

    def process_document_pages(self, doc_paths: List[str]) -> None:
        """Process document pages and generate embeddings"""
        print(f"Start processing {len(doc_paths)} documents...")

        all_page_texts = []
        all_page_index_paths = []

        for doc_path in tqdm(doc_paths, desc="Aggregating page texts"):
            main_text_files = sorted(glob(os.path.join(doc_path, 'text', '*.txt')),
                                     key=lambda x: int(os.path.basename(x).split('.')[0]))
            for page_file in main_text_files:
                try:
                    page_number = os.path.basename(page_file).split('.')[0]
                    with open(page_file, 'r', encoding='utf-8', errors='ignore') as f:
                        main_text = f.read().strip()
                    table_text = ""
                    table_files = glob(os.path.join(doc_path, 'table_text', f'{page_number}_*.txt'))
                    for t_file in table_files:
                        with open(t_file, 'r', encoding='utf-8', errors='ignore') as f:
                            table_text += f.read().strip() + "\n"
                    figure_text = ""
                    figure_files = glob(os.path.join(doc_path, 'figure_text', f'{page_number}_*.txt'))
                    for f_file in figure_files:
                        with open(f_file, 'r', encoding='utf-8', errors='ignore') as f:
                            figure_text += f.read().strip() + "\n"
                    combined_text = f"{main_text}\n\n{table_text}\n\n{figure_text}".strip()
                    if not combined_text:
                        combined_text = " "
                    page_index_path = f"{doc_path}/{page_number}"
                    all_page_texts.append(combined_text)
                    all_page_index_paths.append(page_index_path)
                except Exception as e:
                    print(f"Error processing page {doc_path}/{page_number}: {e}")

        print(f"Aggregated {len(all_page_texts)} pages, start batch embedding...")
        all_embeddings = self.model.encode(
            all_page_texts,
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        for i, page_path in enumerate(all_page_index_paths):
            current_id = len(self.path_to_id)
            self.path_to_id[page_path] = current_id
            self.id_to_path[current_id] = page_path
            self.embeddings.append(all_embeddings[i])

    def save_to_disk(self, output_dir: str) -> None:
        """Save embeddings and mappings to disk"""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'path_to_id.pkl'), 'wb') as f:
            pickle.dump(self.path_to_id, f)
        with open(os.path.join(output_dir, 'id_to_path.pkl'), 'wb') as f:
            pickle.dump(self.id_to_path, f)

        if len(self.embeddings) > 0:
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            vector_dim = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(vector_dim)
            normalized_embeddings = embeddings_array.copy()
            faiss.normalize_L2(normalized_embeddings)
            index.add(normalized_embeddings)
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
        """Search for most similar document pages by text"""
        if self.index is None:
            raise ValueError("Index not loaded. Please call load_from_disk first.")
        if not query_text or not query_text.strip():
            return []
        try:
            query_embedding = self.model.encode(query_text, prompt_name="query")
            query_norm = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_norm)
            actual_top_k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_norm, actual_top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and 0 <= idx < len(self.id_to_path):
                    page_path = self.id_to_path[idx]
                    similarity_score = float(similarities[0][i])
                    results.append((page_path, similarity_score))
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []


# --- Wrapper Functions ---

def embed_and_index_documents_qwen3(ocr_dir: str, output_dir: str, device: str = "cuda"):
    """Embed all document pages and create index with Qwen3"""
    system = DocumentPageEmbeddingSystemQwen3(device=device)
    doc_paths = system.scan_ocr_documents(ocr_dir)
    print(f"Found {len(doc_paths)} document paths in OCR directory")
    system.process_document_pages(doc_paths)
    system.save_to_disk(output_dir)


def search_documents_qwen3(model_dir: str, query_text: str, top_k: int, device: str):
    system = DocumentPageEmbeddingSystemQwen3(device=device)
    system.load_from_disk(model_dir)
    results = system.search_by_text(query_text, top_k=top_k)
    print(f"\nQuery: '{query_text}'")
    print(f"Top {len(results)} results found:")
    for i, (path, score) in enumerate(results):
        print(f"  {i + 1}. Path: {path} (Similarity: {score:.4f})")


def process_json_file_qwen3(json_file_path: str, output_json_path: str, model_dir: str, top_k: int, device: str):
    system = DocumentPageEmbeddingSystemQwen3(device=device)
    system.load_from_disk(model_dir)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Processing {len(data)} JSON entries ...")
    for item in tqdm(data, desc="Processing JSON"):
        try:
            question = item.get("question") or item.get("final_question")
            if not question:
                continue
            results = system.search_by_text(question, top_k=top_k)
            item["retrieval_pages"] = [path for path, _ in results]
        except Exception as e:
            print(f"Error processing entry: {item}. Error: {e}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Processing finished. Results saved to {output_json_path}")


# --- Main Execution ---

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Qwen3 Document Page Embedding System (Multi-GPU)')
    parser.add_argument('--mode', type=str, choices=['embed', 'search', 'process_json'], required=True,
                        help='Mode: embed (embedding and indexing), search (interactive search), process_json (batch process JSON)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Base device, will be overridden by device_map for multi-GPU')
    parser.add_argument('--output_dir', type=str,
                        help='[embed mode] Output directory for index, or [search/process_json mode] directory for loading index')
    parser.add_argument('--ocr_dir', type=str, help='[embed mode] OCR directory containing language/document structure')
    parser.add_argument('--query', type=str, help='[search mode] Query text')
    parser.add_argument('--json_file', type=str, help='[process_json mode] Input JSON file')
    parser.add_argument('--output_json', type=str, help='[process_json mode] Output JSON file')
    parser.add_argument('--top_k', type=int, default=10, help='Number of most similar results to return')
    args = parser.parse_args()

    if args.mode == 'embed':
        if not all([args.ocr_dir, args.output_dir]):
            parser.error("Embed mode (--mode embed) requires --ocr_dir and --output_dir arguments")
        embed_and_index_documents_qwen3(args.ocr_dir, args.output_dir, args.device)
    elif args.mode == 'search':
        if not all([args.output_dir, args.query]):
            parser.error("Search mode (--mode search) requires --output_dir and --query arguments")
        search_documents_qwen3(args.output_dir, args.query, args.top_k, args.device)
    elif args.mode == 'process_json':
        if not all([args.output_dir, args.json_file, args.output_json]):
            parser.error(
                "Process_json mode (--mode process_json) requires --output_dir, --json_file, and --output_json arguments")
        process_json_file_qwen3(args.json_file, args.output_json, args.output_dir, args.top_k, args.device)


if __name__ == "__main__":
    main()