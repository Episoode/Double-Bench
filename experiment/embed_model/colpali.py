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
import json
from colpali_engine.models import ColPali, ColPaliProcessor


class ImageEmbeddingSystem:
    def __init__(
            self,
            model_name: str = "/path/to/your/colpali-v1.3-merged",#vidore/colpali-v1.3-merged
            device: str = "cuda:0",
            batch_size: int = 16
    ):
        """Initialize the image embedding system"""
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

        self.processor = ColPaliProcessor.from_pretrained(model_name, use_fast=True)

        # Store mapping dictionaries
        self.path_to_id = {}
        self.id_to_path = {}
        # Store raw embeddings
        self.original_embeddings = []

    def read_folder_paths(self, txt_path: str) -> List[str]:
        """Read folder paths from txt file"""
        with open(txt_path, 'r') as f:
            folder_paths = [line.strip() for line in f if line.strip()]
        return folder_paths

    def get_image_paths(self, folder_paths: List[str]) -> List[str]:
        """Get image paths from all folders"""
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

            batch_images = self.processor.process_images(images).to(self.model.device)
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
        print(f"Indexed {len(self.path_to_id)} images")

    def load_from_disk(self, input_dir: str) -> None:
        """Load embeddings and mappings from disk"""
        with open(os.path.join(input_dir, 'path_to_id.pkl'), 'rb') as f:
            self.path_to_id = pickle.load(f)
        with open(os.path.join(input_dir, 'id_to_path.pkl'), 'rb') as f:
            self.id_to_path = pickle.load(f)
        self.original_embeddings = torch.load(os.path.join(input_dir, 'original_embeddings.pt'))
        print(f"Loaded embeddings for {len(self.path_to_id)} images from {input_dir}")

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search the most similar images by text"""
        if not self.original_embeddings:
            raise ValueError("Image embeddings are not loaded or empty.")
        batch_query_processed = self.processor.process_queries([query_text])
        for key, value in batch_query_processed.items():
            if isinstance(value, torch.Tensor):
                batch_query_processed[key] = value.to(self.device)
        with torch.no_grad():
            query_output = self.model(**batch_query_processed)

        if hasattr(query_output, 'embeddings') and isinstance(query_output.embeddings, torch.Tensor):
            query_embedding_batched = query_output.embeddings
        elif isinstance(query_output, torch.Tensor):
            query_embedding_batched = query_output
        else:
            raise TypeError(f"Cannot get embedding tensor from model output. Model output type: {type(query_output)}.")

        if query_embedding_batched.ndim == 3 and query_embedding_batched.shape[0] == 1:
            query_embedding_2d = query_embedding_batched[0]
        elif query_embedding_batched.ndim == 2:
            query_embedding_2d = query_embedding_batched
        else:
            raise ValueError(f"Query embedding shape is not as expected: {query_embedding_batched.shape}")

        qs_list = [query_embedding_2d]
        ps_list = self.original_embeddings

        all_scores_matrix = self.processor.score_multi_vector(qs=qs_list, ps=ps_list)
        scores_for_query = all_scores_matrix[0]

        scores_with_indices = sorted(enumerate(scores_for_query.tolist()), key=lambda x: x[1], reverse=True)

        top_results = []
        for i, score in scores_with_indices[:top_k]:
            if i in self.id_to_path:
                top_results.append((self.id_to_path[i], score))
        return top_results


# --- Wrapper Functions ---

def embed_and_index_images(txt_path: str, output_dir: str, device: str):
    system = ImageEmbeddingSystem(device=device)
    folder_paths = system.read_folder_paths(txt_path)
    print(f"Read {len(folder_paths)} folder paths from {txt_path}")
    image_paths = system.get_image_paths(folder_paths)
    print(f"Found {len(image_paths)} images")
    system.batch_process_images(image_paths)
    system.save_to_disk(output_dir)


def search_images(model_dir: str, query_text: str, top_k: int, device: str):
    system = ImageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)
    results = system.search_by_text(query_text, top_k=top_k)
    print(f"Query: '{query_text}'")
    print(f"Found {len(results)} results:")
    for i, (path, score) in enumerate(results):
        print(f"{i + 1}. Path: {path} (Similarity: {score:.4f})")


def process_json_file(json_file_path: str, output_json_path: str, model_dir: str, top_k: int, device: str):
    """Process JSON file, retrieve related images for each question and add to JSON"""
    system = ImageEmbeddingSystem(device=device)
    system.load_from_disk(model_dir)

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Start processing {len(data)} JSON entries...")

    for item in tqdm(data, desc="Processing JSON"):
        question = item.get("question") or item.get("final_question")
        if question:
            results = system.search_by_text(question, top_k=top_k)
            retrieval_pages = [path for path, _ in results]
            item["retrieval_pages"] = retrieval_pages

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Processing finished. Results saved to {output_json_path}")


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Image Embedding and Retrieval System')

    parser.add_argument('--mode', type=str, choices=['embed', 'search', 'process_json'], required=True,
                        help='Mode: embed (embedding and indexing), search (interactive search), process_json (batch process JSON)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--output_dir', type=str,
                        help='[embed mode] Output directory for index, or [search/process_json mode] directory for loading index')
    # Args for 'embed' mode
    parser.add_argument('--txt_path', type=str, help='[embed mode] txt file containing image folder paths')
    # Args for 'search' mode
    parser.add_argument('--query', type=str, help='[search mode] Query text')
    # Args for 'process_json' mode
    parser.add_argument('--json_file', type=str, help='[process_json mode] Input JSON file')
    parser.add_argument('--output_json', type=str, help='[process_json mode] Output JSON file')
    # General args
    parser.add_argument('--top_k', type=int, default=10, help='Number of most similar results to return')

    args = parser.parse_args()

    if args.mode == 'embed':
        if not all([args.txt_path, args.output_dir]):
            parser.error("Embed mode (--mode embed) requires --txt_path and --output_dir arguments.")
        embed_and_index_images(args.txt_path, args.output_dir, args.device)
    elif args.mode == 'search':
        if not all([args.output_dir, args.query]):
            parser.error("Search mode (--mode search) requires --output_dir and --query arguments.")
        search_images(args.output_dir, args.query, args.top_k, args.device)
    elif args.mode == 'process_json':
        if not all([args.output_dir, args.json_file, args.output_json]):
            parser.error("Process_json mode (--mode process_json) requires --output_dir, --json_file, and --output_json arguments.")
        process_json_file(args.json_file, args.output_json, args.output_dir, args.top_k, args.device)


if __name__ == "__main__":
    main()