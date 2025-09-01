import os
import torch
import faiss
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from glob import glob
import argparse
# Ensure the required custom library is in your environment
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class ImageEmbeddingSystem:
    def __init__(
            self,
            model_name: str = "tsystems/colqwen2.5-3b-multilingual-v1.0",
            device: str = "cuda",
            cache_dir: str = "/path/to/your/model/cache/",
            batch_size: int = 2
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = batch_size

        print(f"Loading model from {model_name} and allocating to all available GPUs...")

        self.model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir
        ).eval()

        self.processor = ColQwen2_5_Processor.from_pretrained(
            model_name,
            use_fast=True
        )

        self.path_to_id = {}
        self.id_to_path = {}
        self.original_embeddings = []

        print("Model loaded on multiple GPUs!")
        print("Device map:", self.model.hf_device_map)

    def read_folder_paths(self, txt_path: str) -> List[str]:
        """Read folder paths from a txt file"""
        with open(txt_path, 'r') as f:
            folder_paths = [line.strip() for line in f if line.strip()]
        return folder_paths

    def get_image_paths(self, folder_paths: List[str]) -> List[str]:
        """Get all image paths from the folders"""
        image_paths = []
        for folder in folder_paths:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_paths.extend(glob(os.path.join(folder, ext)))
        return image_paths

    def batch_process_images(self, image_paths: List[str]) -> None:
        """Batch process images to generate embeddings"""
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Embedding Images"):
            batch_paths = image_paths[i:i + self.batch_size]
            images = []
            valid_paths = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")

            if not images:
                continue

            batch_images = self.processor.process_images(images)

            with torch.no_grad():
                image_embeddings = self.model(**batch_images)

            for j, img_path in enumerate(valid_paths):
                current_id = len(self.path_to_id)
                self.path_to_id[img_path] = current_id
                self.id_to_path[current_id] = img_path
                single_embedding = image_embeddings[j].cpu()
                self.original_embeddings.append(single_embedding)

    def save_to_disk(self, output_dir: str) -> None:
        """Save embeddings and mappings to disk"""
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'path_to_id.pkl'), 'wb') as f:
            pickle.dump(self.path_to_id, f)

        with open(os.path.join(output_dir, 'id_to_path.pkl'), 'wb') as f:
            pickle.dump(self.id_to_path, f)

        torch.save(self.original_embeddings, os.path.join(output_dir, 'original_embeddings.pt'))

        if self.original_embeddings:
            flattened_dim = self.original_embeddings[0].numel()
            index = faiss.IndexFlatL2(flattened_dim)

            for embedding in self.original_embeddings:
                flat_embedding = embedding.reshape(1, -1).half().numpy()
                index.add(flat_embedding)

            faiss.write_index(index, os.path.join(output_dir, 'vector.index'))

        print(f"All data saved to: {output_dir}")
        print(f"Total indexed images: {len(self.path_to_id)}")

    def load_from_disk(self, input_dir: str) -> None:
        """Load embeddings and mappings from disk"""
        with open(os.path.join(input_dir, 'path_to_id.pkl'), 'rb') as f:
            self.path_to_id = pickle.load(f)

        with open(os.path.join(input_dir, 'id_to_path.pkl'), 'rb') as f:
            self.id_to_path = pickle.load(f)

        self.original_embeddings = torch.load(os.path.join(input_dir, 'original_embeddings.pt'))

        print(f"Loaded embeddings for {len(self.path_to_id)} images from {input_dir}")

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for the most similar images by text"""
        if not self.original_embeddings:
            raise ValueError("Image embeddings are not loaded or empty.")

        batch_query_processed = self.processor.process_queries([query_text])

        with torch.no_grad():
            query_output = self.model(**batch_query_processed)

        if hasattr(query_output, 'embeddings') and isinstance(query_output.embeddings, torch.Tensor):
            query_embedding_batched = query_output.embeddings
        elif isinstance(query_output, torch.Tensor):
            query_embedding_batched = query_output
        else:
            raise TypeError(f"Cannot extract embedding tensor from model output. Model output type: {type(query_output)}.")

        if query_embedding_batched.ndim == 3 and query_embedding_batched.shape[0] == 1:
            query_embedding_2d = query_embedding_batched[0]
        elif query_embedding_batched.ndim == 2:
            query_embedding_2d = query_embedding_batched
        else:
            raise ValueError(f"Query embedding shape is not as expected: {query_embedding_batched.shape}")

        qs_list = [query_embedding_2d]
        ps_list = self.original_embeddings

        if not ps_list:
            raise ValueError("ps_list (self.original_embeddings) is empty.")

        if not all(isinstance(t, torch.Tensor) and t.ndim == 2 for t in ps_list):
            print("Warning: one or more elements in ps_list (self.original_embeddings) may not be the expected 2D tensor.")

        all_scores_matrix = self.processor.score_multi_vector(qs=qs_list, ps=ps_list)
        scores_for_query = all_scores_matrix[0]

        scores_with_indices = sorted(enumerate(scores_for_query.tolist()), key=lambda x: x[1], reverse=True)

        top_results = []
        for i, score in scores_with_indices[:top_k]:
            if i in self.id_to_path:
                top_results.append((self.id_to_path[i], score))
        return top_results


def embed_and_index_images(txt_path: str, output_dir: str, device: str):
    """Read folder paths from txt, embed all images and create index"""
    system = ImageEmbeddingSystem(device=device)
    folder_paths = system.read_folder_paths(txt_path)
    print(f"Read {len(folder_paths)} folder paths from {txt_path}")
    image_paths = system.get_image_paths(folder_paths)
    print(f"Found {len(image_paths)} images")
    system.batch_process_images(image_paths)
    system.save_to_disk(output_dir)


def search_images(model_dir: str, query_text: str, top_k: int, device: str):
    """Load existing index and search images"""
    system = ImageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)
    results = system.search_by_text(query_text, top_k=top_k)
    print(f"Query: '{query_text}'")
    print(f"Found {len(results)} results:")
    for i, (path, score) in enumerate(results):
        print(f"{i + 1}. Path: {path}")
        print(f"   Similarity score: {score:.4f}")


def main():
    """Main function handling command line arguments"""
    parser = argparse.ArgumentParser(description='Image Embedding and Retrieval System (Multi-GPU)')
    parser.add_argument('--txt_path', type=str, required=True, help='Txt file containing folder paths')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Base device. When using multi-GPU, this is overridden by device_map')
    parser.add_argument('--mode', type=str, choices=['embed', 'search'], default='embed',
                        help='Mode: embed (embedding and indexing) or search (search)')
    parser.add_argument('--query', type=str, help='Query text for search mode')
    parser.add_argument('--top_k', type=int, default=5, help='Number of most similar results to return')

    args = parser.parse_args()

    if args.mode == 'embed':
        embed_and_index_images(args.txt_path, args.output_dir, args.device)
    elif args.mode == 'search':
        if not args.query:
            parser.error("Search mode requires --query parameter")
        search_images(args.output_dir, args.query, args.top_k, args.device)


if __name__ == "__main__":
    main()