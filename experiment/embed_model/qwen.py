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
            model_name: str = "/path/to/your/Qwen3-Embedding-4B",#Qwen/Qwen3-Embedding-4B
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
        print("Model loaded with multi-GPU support!")

    def read_folder_paths(self, txt_path: str) -> List[str]:
        """Read folder paths from txt file"""
        with open(txt_path, 'r') as f:
            folder_paths = [line.strip() for line in f if line.strip()]
        return folder_paths

    def process_document_pages(self, doc_paths: List[str]) -> None:
        """Process document pages and generate embeddings"""
        print(f"Start processing {len(doc_paths)} documents...")

        all_page_texts = []
        all_page_index_paths = []

        for doc_path in tqdm(doc_paths, desc="Aggregating page texts"):
            ocr_path = doc_path.replace('DOC3000', 'OCR3000')
            main_text_files = sorted(glob(os.path.join(ocr_path, 'text', '*.txt')),
                                     key=lambda x: int(os.path.basename(x).split('.')[0]))
            for page_file in main_text_files:
                try:
                    page_number = os.path.basename(page_file).split('.')[0]
                    with open(page_file, 'r', encoding='utf-8', errors='ignore') as f:
                        main_text = f.read().strip()
                    table_text = ""
                    table_files = glob(os.path.join(ocr_path, 'table_text', f'{page_number}_*.txt'))
                    for t_file in table_files:
                        with open(t_file, 'r', encoding='utf-8', errors='ignore') as f:
                            table_text += f.read().strip() + "\n"
                    figure_text = ""
                    figure_files = glob(os.path.join(ocr_path, 'figure_text', f'{page_number}_*.txt'))
                    for f_file in figure_files:
                        with open(f_file, 'r', encoding='utf-8', errors='ignore') as f:
                            figure_text += f.read().strip() + "\n"
                    combined_text = f"{main_text}\n\n{table_text}\n\n{figure_text}".strip()
                    if not combined_text:
                        combined_text = " "
                    page_index_path = f"{ocr_path}/{page_number}"
                    all_page_texts.append(combined_text)
                    all_page_index_paths.append(page_index_path)
                except Exception as e:
                    print(f"Error processing page {ocr_path}/{page_number}: {e}")

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
        embeddings_array = np.array(self.embeddings, dtype=np.float32)
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings_array)
        if len(self.embeddings) > 0:
            vector_dim = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(vector_dim)
            normalized_embeddings = embeddings_array.copy()
            faiss.normalize_L2(normalized_embeddings)
            index.add(normalized_embeddings)
            faiss.write_index(index, os.path.join(output_dir, 'vector.index'))
        print(f"All data saved to: {output_dir}")
        print(f"Total indexed document pages: {len(self.path_to_id)}")

    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for most similar document pages by text"""
        if not query_text or not query_text.strip():
            return []
        try:
            query_embedding = self.model.encode(query_text, prompt_name="query")
            index = faiss.IndexFlatIP(len(query_embedding))
            normalized_embeddings = np.array(self.embeddings, dtype=np.float32)
            faiss.normalize_L2(normalized_embeddings)
            index.add(normalized_embeddings)
            query_norm = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_norm)
            actual_top_k = min(top_k, len(self.embeddings))
            similarities, indices = index.search(query_norm, actual_top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.id_to_path):
                    page_path = self.id_to_path[idx]
                    similarity_score = float(similarities[0][i])
                    results.append((page_path, similarity_score))
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

def embed_and_index_documents_qwen3(txt_path: str, output_dir: str, device: str = "cuda"):
    """Embed all document pages and create index with Qwen3"""
    system = DocumentPageEmbeddingSystemQwen3(device=device)
    doc_paths = system.read_folder_paths(txt_path)
    print(f"Read {len(doc_paths)} document paths from {txt_path}")
    system.process_document_pages(doc_paths)
    system.save_to_disk(output_dir)
    return system

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Qwen3 Document Page Embedding System (Multi-GPU)')
    parser.add_argument('--txt_path', type=str, required=True, help='Txt file containing document paths')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Base device, will be overridden by device_map for multi-GPU')
    args = parser.parse_args()
    embed_and_index_documents_qwen3(
        txt_path=args.txt_path,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == "__main__":
    main()