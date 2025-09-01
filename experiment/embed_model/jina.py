import os
import torch
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from typing import List, Tuple, Dict
from glob import glob
import argparse
from sentence_transformers import SentenceTransformer

class JinaImageEmbeddingSystem:
    def __init__(
            self,
            model_name: str = "/path/to/your/jina-embeddings-v4",#jinaai/jina-embeddings-v4
            device: str = "cuda",
            batch_size: int = 32,
    ):
        """Initialize the Jina V4-based image embedding system"""
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        print(f"Loading model: {self.model_name}...")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,  # Required for Jina V4
            device=device,
            model_kwargs={
                'attn_implementation': 'flash_attention_2',
                'torch_dtype': torch.float16
            }
        )
        if 'cuda' in self.device:
            self.model.half()
        print("Model loaded.")

        self.path_to_id: Dict[str, int] = {}
        self.id_to_path: Dict[int, str] = {}
        self.embeddings: List[np.ndarray] = []

        self.index = None

    def read_folder_paths(self, txt_path: str) -> List[str]:
        """Read folder paths from txt file"""
        with open(txt_path, 'r') as f:
            folder_paths = [line.strip() for line in f if line.strip()]
        return folder_paths

    def get_image_paths(self, folder_paths: List[str]) -> List[str]:
        """Get all jpg image paths from the folders"""
        image_paths = []
        for folder in folder_paths:
            jpg_files = glob(os.path.join(folder, "*.jpg"))
            image_paths.extend(jpg_files)
        return image_paths

    def batch_process_images(self, image_paths: List[str]) -> None:
        """
        Batch process images using SentenceTransformer and generate embeddings.
        The encode function can directly handle image paths.
        """
        print(f"Generating embeddings for {len(image_paths)} images using {self.model_name} ...")
        all_embeddings = self.model.encode(
            sentences=image_paths,
            batch_size=self.batch_size,
            task="retrieval",
            show_progress_bar=True
        )

        for i, img_path in enumerate(image_paths):
            self.path_to_id[img_path] = i
            self.id_to_path[i] = img_path

        self.embeddings = list(all_embeddings)

    def save_to_disk(self, output_dir: str) -> None:
        """Save embeddings, mappings, and FAISS index to disk"""
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'path_to_id.pkl'), 'wb') as f:
            pickle.dump(self.path_to_id, f)
        with open(os.path.join(output_dir, 'id_to_path.pkl'), 'wb') as f:
            pickle.dump(self.id_to_path, f)

        if self.embeddings:
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            vector_dim = embeddings_array.shape[1]

            index = faiss.IndexFlatIP(vector_dim)
            faiss.normalize_L2(embeddings_array)
            index.add(embeddings_array)
            faiss.write_index(index, os.path.join(output_dir, 'vector.index'))
            print(f"FAISS index created and saved. Vector dimension: {vector_dim}")

        print(f"All data saved to: {output_dir}")
        print(f"Total images indexed: {len(self.path_to_id)}")

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
            print(f"Loaded FAISS index for {self.index.ntotal} images from {input_dir}.")
        else:
            raise FileNotFoundError("Error: vector.index file not found in the specified directory.")

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Efficiently search for the most similar images using FAISS index by text.
        """
        if self.index is None:
            raise ValueError("FAISS index not loaded. Please call load_from_disk first.")

        query_embedding = self.model.encode(
            sentences=[query_text],
            task="retrieval",
            prompt_name="query"  # Use prompt optimized for retrieval
        )

        query_embedding_norm = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding_norm)

        distances, indices = self.index.search(query_embedding_norm, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                image_path = self.id_to_path[idx]
                score = distances[0][i]
                results.append((image_path, score))

        return results

def embed_and_index_images(txt_path: str, output_dir: str, device: str = "cuda:0"):
    """Main workflow: read folder paths from txt, embed all images, and create index"""
    system = JinaImageEmbeddingSystem(device=device)
    folder_paths = system.read_folder_paths(txt_path)
    print(f"Read {len(folder_paths)} folder paths from {txt_path}.")
    image_paths = system.get_image_paths(folder_paths)
    print(f"Found {len(image_paths)} JPG images in these paths.")

    system.batch_process_images(image_paths)
    system.save_to_disk(output_dir)
    return system

def search_images(model_dir: str, query_text: str, top_k: int = 10, device: str = "cuda"):
    """Main workflow: load existing index and search images"""
    system = JinaImageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)

    results = system.search_by_text(query_text, top_k=top_k)

    print(f"\nQuery: '{query_text}'")
    print(f"Top {len(results)} results found:")
    for i, (path, score) in enumerate(results):
        print(f"  {i + 1}. Path: {path}")
        print(f"     Similarity score: {score:.4f}")

    return results

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Jina V4-based Image Embedding and Retrieval System')
    parser.add_argument('--txt_path', type=str, required=True, help='Txt file containing folder paths')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for index and mappings')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--mode', type=str, choices=['embed', 'search'], required=True,
                        help='Mode: embed (embedding and indexing) or search (search)')
    parser.add_argument('--query', type=str, help='Query text for search mode')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return in search mode')

    args = parser.parse_args()

    if args.mode == 'embed':
        embed_and_index_images(args.txt_path, args.output_dir, args.device)
    elif args.mode == 'search':
        if not args.query:
            parser.error("Search mode (--mode search) requires --query argument")
        search_images(args.output_dir, args.query, args.top_k, args.device)

if __name__ == "__main__":
    main()