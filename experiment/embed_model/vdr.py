import torch
import os
import numpy as np
import faiss
import pickle
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Dict
from glob import glob
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import argparse

class ImageEmbeddingSystem:
    def __init__(
            self,
            model_name: str = "llamaindex/vdr-2b-multi-v1",
            device: str = "cuda",
            cache_folder: str = "/path/to/your/model/cache/",
            batch_size: int = 128
    ):
        """Initialize image embedding system"""
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.batch_size = batch_size

        # Load model
        self.model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            trust_remote_code=True,
            cache_folder=cache_folder
        )

        self.path_to_id: Dict[str, int] = {}
        self.id_to_path: Dict[int, str] = {}
        self.embeddings: List[List[float]] = []

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

    def process_images(self, image_paths: List[str]) -> None:
        """Process images and generate embeddings"""
        for img_path in tqdm(image_paths):
            try:
                with open(img_path, 'rb'):
                    pass
                embedding = self.model.get_image_embedding(img_path)
                current_id = len(self.path_to_id)
                self.path_to_id[img_path] = current_id
                self.id_to_path[current_id] = img_path
                self.embeddings.append(embedding)
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")

    def save_to_disk(self, output_dir: str) -> None:
        """Save embeddings and mappings to disk"""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'path_to_id.pkl'), 'wb') as f:
            pickle.dump(self.path_to_id, f)
        with open(os.path.join(output_dir, 'id_to_path.pkl'), 'wb') as f:
            pickle.dump(self.id_to_path, f)
        embeddings_array = np.array(self.embeddings, dtype=np.float32)
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings_array)
        if self.embeddings:
            vector_dim = len(self.embeddings[0])
            index = faiss.IndexFlatIP(vector_dim)
            normalized_embeddings = embeddings_array.copy()
            faiss.normalize_L2(normalized_embeddings)
            index.add(normalized_embeddings)
            faiss.write_index(index, os.path.join(output_dir, 'vector.index'))
        print(f"All data saved to: {output_dir}")
        print(f"Total images indexed: {len(self.path_to_id)}")

    def load_from_disk(self, input_dir: str) -> None:
        """Load embeddings and mappings from disk"""
        with open(os.path.join(input_dir, 'path_to_id.pkl'), 'rb') as f:
            self.path_to_id = pickle.load(f)
        with open(os.path.join(input_dir, 'id_to_path.pkl'), 'rb') as f:
            self.id_to_path = pickle.load(f)
        self.embeddings = np.load(os.path.join(input_dir, 'embeddings.npy')).tolist()
        print(f"Loaded embeddings for {len(self.path_to_id)} images from {input_dir}")

    def batch_cosine_similarity(self, query_vec: List[float], all_vecs: List[List[float]]) -> List[float]:
        """Batch compute cosine similarity between query vector and all image vectors"""
        query_tensor = torch.tensor(query_vec, dtype=torch.bfloat16).to(self.device)
        all_tensors = torch.tensor(all_vecs, dtype=torch.bfloat16).to(self.device)
        query_norm = torch.nn.functional.normalize(query_tensor.unsqueeze(0), p=2, dim=1)
        all_norm = torch.nn.functional.normalize(all_tensors, p=2, dim=1)
        similarities = torch.mm(query_norm, all_norm.t()).squeeze()
        return similarities.cpu().tolist()

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search most similar images by text"""
        if not query_text or not query_text.strip():
            return []
        try:
            query_embedding = self.model.get_query_embedding(query_text)
            if not self.embeddings or len(self.embeddings) == 0:
                print("Warning: No available image embeddings")
                return []
            similarities = self.batch_cosine_similarity(query_embedding, self.embeddings)
            actual_top_k = min(top_k, len(similarities))
            top_indices = np.argsort(similarities)[::-1][:actual_top_k]
            results = []
            for idx in top_indices:
                int_idx = int(idx)
                image_path = self.id_to_path[int_idx]
                similarity_score = similarities[int_idx]
                results.append((image_path, float(similarity_score)))
            return results
        except Exception as e:
            print(f"Error processing query '{query_text}': {e}")
            return []

def embed_and_index_images(txt_path: str, output_dir: str, device: str = "cuda"):
    """Read folder paths from txt, embed all images and create index"""
    system = ImageEmbeddingSystem(device=device)
    folder_paths = system.read_folder_paths(txt_path)
    print(f"Read {len(folder_paths)} folder paths from {txt_path}")
    image_paths = system.get_image_paths(folder_paths)
    print(f"Found {len(image_paths)} JPG images")
    system.process_images(image_paths)
    system.save_to_disk(output_dir)
    return system

def search_images(model_dir: str, query_text: str, top_k: int = 5, device: str = "cuda"):
    """Load existing index and search images"""
    system = ImageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)
    results = system.search_by_text(query_text, top_k=top_k)
    print(f"Query: '{query_text}'")
    print(f"Found {len(results)} results:")
    for i, (path, score) in enumerate(results):
        print(f"{i + 1}. Path: {path}")
        print(f"   Similarity score: {score:.4f}")
    return results

def main():
    """Main function for argument parsing"""
    parser = argparse.ArgumentParser(description='Image Embedding and Retrieval System')
    parser.add_argument('--mode', type=str, required=True, choices=['embed', 'search'],
                        help='Operation mode: embed (embedding and indexing) or search (search)')
    parser.add_argument('--txt_path', type=str, help='Txt file containing folder paths')
    parser.add_argument('--output_dir', type=str, required=True, help='Output/index directory')
    parser.add_argument('--query', type=str, help='Query text for search mode')
    parser.add_argument('--top_k', type=int, default=5, help='Number of most similar results to return')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g. cuda, cuda:0, cuda:1)')

    args = parser.parse_args()

    if args.mode == 'embed':
        if not args.txt_path:
            parser.error("Embed mode requires --txt_path argument")
        embed_and_index_images(args.txt_path, args.output_dir, args.device)
    elif args.mode == 'search':
        if not args.query:
            parser.error("Search mode requires --query argument")
        search_images(args.output_dir, args.query, args.top_k, args.device)

if __name__ == "__main__":
    main()