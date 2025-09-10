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

def add_eos(input_examples: List[str], tokenizer) -> List[str]:
    """Add eos_token for each input text."""
    return [input_example + tokenizer.eos_token for input_example in input_examples]

class DocumentPageEmbeddingSystem:
    def __init__(
            self,
            model_name: str = "nvidia/NV-Embed-v2",
            device: str = "cuda",
            batch_size: int = 2
    ):
        """Initialize document page embedding system."""
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"

        task_instruct = "Given a question, retrieve passages that answer the question"
        self.query_prefix = f"Instruct: {task_instruct}\nQuery: "

        self.path_to_id: Dict[str, int] = {}
        self.id_to_path: Dict[int, str] = {}
        self.embeddings: List[np.ndarray] = []
        self.index = None

    def get_document_paths(self, ocr_dir: str) -> List[str]:
        """Get all document paths from OCR directory structure"""
        doc_paths = []
        for lang_dir in os.listdir(ocr_dir):
            lang_path = os.path.join(ocr_dir, lang_dir)
            if os.path.isdir(lang_path):
                for doc_name in os.listdir(lang_path):
                    doc_path = os.path.join(lang_path, doc_name)
                    if os.path.isdir(doc_path):
                        doc_paths.append(doc_path)
        return doc_paths

    def process_document_pages(self, doc_paths: List[str]) -> None:
        """Process document pages and generate embeddings"""
        all_page_texts, all_page_index_paths = [], []
        for doc_path in tqdm(doc_paths, desc="Processing documents"):
            main_text_files = sorted(glob(os.path.join(doc_path, 'text', '*.txt')),
                                     key=lambda x: int(os.path.basename(x).split('.')[0]))
            for page_file in main_text_files:
                try:
                    page_number = os.path.basename(page_file).split('.')[0]
                    with open(page_file, 'r', encoding='utf-8', errors='ignore') as f:
                        main_text = f.read().strip()
                    table_text = ""
                    for t_file in glob(os.path.join(doc_path, 'table_text', f'{page_number}_*.txt')):
                        with open(t_file, 'r', encoding='utf-8', errors='ignore') as f:
                            table_text += f.read().strip() + "\n"
                    figure_text = ""
                    for f_file in glob(os.path.join(doc_path, 'figure_text', f'{page_number}_*.txt')):
                        with open(f_file, 'r', encoding='utf-8', errors='ignore') as f:
                            figure_text += f.read().strip() + "\n"

                    combined_text = f"{main_text}\n\n{table_text}\n\n{figure_text}".strip() or " "
                    page_index_path = f"{doc_path}/{page_number}"
                    all_page_texts.append(combined_text)
                    all_page_index_paths.append(page_index_path)
                except Exception as e:
                    pass

        if all_page_texts:
            all_embeddings = self.model.encode(
                add_eos(all_page_texts, self.model.tokenizer),
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            for i, page_path in enumerate(all_page_index_paths):
                self.path_to_id[page_path] = i
                self.id_to_path[i] = page_path
                self.embeddings.append(all_embeddings[i])

    def save_to_disk(self, output_dir: str) -> None:
        """Save embeddings and mappings to disk."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'path_to_id.pkl'), 'wb') as f:
            pickle.dump(self.path_to_id, f)
        with open(os.path.join(output_dir, 'id_to_path.pkl'), 'wb') as f:
            pickle.dump(self.id_to_path, f)
        if self.embeddings:
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            vector_dim = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(vector_dim)
            index.add(embeddings_array)
            faiss.write_index(index, os.path.join(output_dir, 'vector.index'))

    def load_from_disk(self, input_dir: str) -> None:
        """Load mappings and FAISS index from disk."""
        with open(os.path.join(input_dir, 'path_to_id.pkl'), 'rb') as f:
            self.path_to_id = pickle.load(f)
        with open(os.path.join(input_dir, 'id_to_path.pkl'), 'rb') as f:
            self.id_to_path = pickle.load(f)
        index_path = os.path.join(input_dir, 'vector.index')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"vector.index not found in {input_dir}")

    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Efficient text search in loaded FAISS index."""
        if self.index is None:
            raise ValueError("Index not loaded. Please call load_from_disk first.")
        if not query_text or not query_text.strip():
            return []
        try:
            query_embedding = self.model.encode(
                add_eos([query_text], self.model.tokenizer),
                prompt=self.query_prefix,
                normalize_embeddings=True
            )
            actual_top_k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_embedding, actual_top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    results.append((self.id_to_path[idx], float(similarities[0][i])))
            return results
        except Exception as e:
            return []

def embed_and_index_documents(ocr_dir: str, output_dir: str, device: str):
    system = DocumentPageEmbeddingSystem(device=device)
    doc_paths = system.get_document_paths(ocr_dir)
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
    for item in tqdm(data, desc="Processing JSON"):
        try:
            question = item.get("question") or item.get("final_question")
            if not question:
                continue
            results = system.search_by_text(question, top_k=top_k)
            item["retrieval_pages"] = [path for path, _ in results]
        except Exception as e:
            pass
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='NV-Embed-v2 Document Page Embedding and Retrieval System')
    parser.add_argument('--mode', type=str, choices=['embed', 'search', 'process_json'], required=True,
                        help='Mode: embed, search, or process_json')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, help='Output directory for index or directory for loading index')
    parser.add_argument('--ocr_dir', type=str, help='[embed mode] OCR directory containing documents')
    parser.add_argument('--query', type=str, help='[search mode] Query text')
    parser.add_argument('--json_file', type=str, help='[process_json mode] Input JSON file')
    parser.add_argument('--output_json', type=str, help='[process_json mode] Output JSON file')
    parser.add_argument('--top_k', type=int, default=10, help='Number of results to return')
    args = parser.parse_args()

    if args.mode == 'embed':
        if not all([args.ocr_dir, args.output_dir]):
            parser.error("Embed mode requires --ocr_dir and --output_dir arguments")
        embed_and_index_documents(args.ocr_dir, args.output_dir, args.device)
    elif args.mode == 'search':
        if not all([args.output_dir, args.query]):
            parser.error("Search mode requires --output_dir and --query arguments")
        search_documents(args.output_dir, args.query, args.top_k, args.device)
    elif args.mode == 'process_json':
        if not all([args.output_dir, args.json_file, args.output_json]):
            parser.error("Process_json mode requires --output_dir, --json_file, and --output_json arguments")
        process_json_file(args.json_file, args.output_json, args.output_dir, args.top_k, args.device)

if __name__ == "__main__":
    main()