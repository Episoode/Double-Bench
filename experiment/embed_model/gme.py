import json
import os
import torch
from tqdm import tqdm
from typing import List, Dict, Union, Tuple
import argparse
import faiss
import numpy as np
import pickle
from glob import glob
from transformers import AutoModel
from transformers.utils.versions import require_version

os.environ["TOKENIZERS_PARALLELISM"] = "false"

require_version(
    "transformers<4.52.0",
    "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)


class MultimodalVectorDB:
    def __init__(
            self,
            model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
            device: str = "cuda:0",
            batch_size: int = 16
    ):
        """Initialize the multimodal vector database."""
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        print("Loading model...")
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype="float16", device_map=device, trust_remote_code=True
        )
        print("Model loaded!")
        self.path_to_id, self.id_to_path, self.id_to_content, self.id_to_type = {}, {}, {}, {}
        self.embeddings = []
        self.t2i_prompt = 'Find an image that matches the given text.'
        self.i2t_prompt = 'Find text that matches the given image.'

    def read_folder_paths(self, txt_path: str) -> List[str]:
        """Read folder paths from txt file."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def get_image_files(self, folder_paths: List[str]) -> List[str]:
        """Get all image files from the folders."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for folder in folder_paths:
            if not os.path.exists(folder):
                continue
            for ext in image_extensions:
                files = glob(os.path.join(folder, f"*{ext.lower()}"))
                files.extend(glob(os.path.join(folder, f"*{ext.upper()}")))
                image_files.extend(files)
        return sorted(image_files)

    def get_text_files(self, folder_paths: List[str]) -> List[str]:
        """Get all txt files from the folders."""
        text_files = []
        for folder in folder_paths:
            if not os.path.exists(folder):
                continue
            txt_files = glob(os.path.join(folder, "*.txt"))

            def extract_number(f):
                try:
                    return int(os.path.basename(f).split('.')[0])
                except ValueError:
                    return float('inf')

            text_files.extend(sorted(txt_files, key=extract_number))
        return text_files

    def _add_item(self, path, content, item_type, embed):
        """Internal method to add an entry to the database."""
        current_id = len(self.path_to_id)
        self.path_to_id[path] = current_id
        self.id_to_path[current_id] = path
        self.id_to_content[current_id] = content
        self.id_to_type[current_id] = item_type
        self.embeddings.append(embed.cpu().numpy())

    def process_images(self, image_files: List[str]) -> None:
        """Process image files and generate embeddings."""
        print(f"Processing {len(image_files)} image files...")
        for i in tqdm(range(0, len(image_files), self.batch_size), desc="Processing images"):
            batch_files = [p for p in image_files[i:i + self.batch_size] if os.path.exists(p)]
            if not batch_files:
                continue
            try:
                embeddings = self.model.get_image_embeddings(images=batch_files, is_query=False)
                for img_path, embed in zip(batch_files, embeddings):
                    self._add_item(img_path, img_path, 'image', embed)
            except Exception as e:
                print(f"Error processing image batch: {e}")

    def process_texts(self, text_files: List[str]) -> None:
        """Process text files and generate embeddings."""
        print(f"Processing {len(text_files)} text files...")
        for i in tqdm(range(0, len(text_files), self.batch_size), desc="Processing texts"):
            batch_files = text_files[i:i + self.batch_size]
            batch_texts_map = {p: open(p, 'r', encoding='utf-8').read().strip() for p in batch_files}
            valid_texts = {p: c for p, c in batch_texts_map.items() if c}
            if not valid_texts:
                continue
            try:
                embeddings = self.model.get_text_embeddings(texts=list(valid_texts.values()))
                for (path, content), embed in zip(valid_texts.items(), embeddings):
                    self._add_item(path, content, 'text', embed)
            except Exception as e:
                print(f"Error processing text batch: {e}")

    def build_database(self, txt_path: str, data_type: str = 'auto'):
        """Build vector database."""
        folder_paths = self.read_folder_paths(txt_path)
        print(f"Read {len(folder_paths)} folder paths from {txt_path}")
        if data_type in ['auto', 'image']:
            self.process_images(self.get_image_files(folder_paths))
        if data_type in ['auto', 'text']:
            self.process_texts(self.get_text_files(folder_paths))

    def save_database(self, output_dir: str):
        """Save vector database to disk."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'path_to_id.pkl'), 'wb') as f:
            pickle.dump(self.path_to_id, f)
        with open(os.path.join(output_dir, 'id_to_path.pkl'), 'wb') as f:
            pickle.dump(self.id_to_path, f)
        with open(os.path.join(output_dir, 'id_to_content.pkl'), 'wb') as f:
            pickle.dump(self.id_to_content, f)
        with open(os.path.join(output_dir, 'id_to_type.pkl'), 'wb') as f:
            pickle.dump(self.id_to_type, f)
        if self.embeddings:
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            vector_dim = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(vector_dim)
            faiss.normalize_L2(embeddings_array)
            index.add(embeddings_array)
            faiss.write_index(index, os.path.join(output_dir, 'vector.index'))
            print(f"Vector database saved to: {output_dir}")
            print(f"Total files indexed: {len(self.path_to_id)}")

    def load_database(self, input_dir: str):
        """Load vector database from disk."""
        with open(os.path.join(input_dir, 'id_to_path.pkl'), 'rb') as f:
            self.id_to_path = pickle.load(f)
        with open(os.path.join(input_dir, 'id_to_content.pkl'), 'rb') as f:
            self.id_to_content = pickle.load(f)
        with open(os.path.join(input_dir, 'id_to_type.pkl'), 'rb') as f:
            self.id_to_type = pickle.load(f)
        self.index = faiss.read_index(os.path.join(input_dir, 'vector.index'))
        print(f"Loaded database from {input_dir} with {self.index.ntotal} vectors.")

    def search(self, query: str, query_type: str = 'text', target_type: str = 'all', top_k: int = 5) -> List[Dict]:
        """Search for similar content."""
        if not query or not query.strip():
            return []
        try:
            if query_type == 'text':
                instruction = self.t2i_prompt if target_type == 'image' else None
                query_embedding = self.model.get_text_embeddings(texts=[query], instruction=instruction)[0].cpu().numpy()
            elif query_type == 'image':
                instruction = self.i2t_prompt if target_type == 'text' else None
                query_embedding = self.model.get_image_embeddings(images=[query], instruction=instruction, is_query=True)[0].cpu().numpy()
            else:
                raise ValueError("query_type must be 'text' or 'image'")

            if not self.index or self.index.ntotal == 0:
                print("Warning: database is empty or not loaded.")
                return []

            query_norm = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_norm)

            if target_type in ['image', 'text']:
                ids_to_search = np.array([i for i, t in self.id_to_type.items() if t == target_type], dtype=np.int64)
                if len(ids_to_search) == 0:
                    return []
                sub_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.index.d))
                sub_index.add_with_ids(self.index.reconstruct_n(0, self.index.ntotal)[ids_to_search], ids_to_search)
                similarities, indices = sub_index.search(query_norm, min(top_k, len(ids_to_search)))
            else:
                similarities, indices = self.index.search(query_norm, min(top_k, self.index.ntotal))

            results = []
            for i, original_idx in enumerate(indices[0]):
                if original_idx < 0:
                    continue
                results.append({
                    'path': self.id_to_path[original_idx],
                    'type': self.id_to_type[original_idx],
                    'similarity': float(similarities[0][i]),
                    'content': self.id_to_content[original_idx]
                })
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []


def process_json_and_save(json_path: str, output_json_path: str, db_dir: str, device: str = "cuda:0"):
    """
    Read a JSON file, perform batch retrieval for each query, and save results.
    """
    db = MultimodalVectorDB(device=device)
    db.load_database(db_dir)

    with open(json_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    print(f"Processing {len(queries)} JSON queries...")
    for query_data in tqdm(queries, desc="Processing JSON queries"):
        query = query_data.get('query') or query_data.get('question') or query_data.get('final_question')
        if not query:
            continue

        query_type = query_data.get('query_type', 'text')
        target_type = query_data.get('target_type', 'all')
        top_k = query_data.get('top_k', 10)

        results = db.search(query, query_type, target_type, top_k)
        query_data['retrieval_results'] = results

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=4, ensure_ascii=False)

    print(f"Processing finished. Results saved to: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description='Multimodal Vector Database System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    build_parser = subparsers.add_parser('build', help='Build the vector database')
    build_parser.add_argument('--txt_path', type=str, required=True, help='Txt file with folder paths')
    build_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    build_parser.add_argument('--data_type', type=str, choices=['auto', 'image', 'text'], default='auto', help='Data type')
    build_parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device')
    build_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    process_parser = subparsers.add_parser('process_json', help='Read JSON, batch retrieve, and save results')
    process_parser.add_argument('--json_path', type=str, required=True, help='Input JSON file with queries')
    process_parser.add_argument('--db_dir', type=str, required=True, help='Database directory')
    process_parser.add_argument('--output_json', type=str, required=True, help='Output JSON file for results')
    process_parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device')

    args = parser.parse_args()

    if args.command == 'build':
        db = MultimodalVectorDB(device=args.device, batch_size=args.batch_size)
        db.build_database(args.txt_path, args.data_type)
        db.save_database(args.output_dir)
    elif args.command == 'process_json':
        process_json_and_save(args.json_path, args.output_json, args.db_dir, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()