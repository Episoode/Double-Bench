import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode, Document, ImageDocument, TextNode
from llama_index.core import SimpleDirectoryReader
from PIL import Image
import base64
from io import BytesIO

from llms.vl_embedding import VL_Embedding

class Ingestion:
    def __init__(self, dataset_dir,input_prefix='ppocr',output_prefix='bge_ingestion',embed_model_name='BAAI/bge-m3'):
        self.dataset_dir = dataset_dir
        self.input_dir  = os.path.join(dataset_dir, input_prefix)
        self.output_dir = os.path.join(dataset_dir, output_prefix)
        self.output_file_format = 'document'
        # self.chunk_size = 1024
        self.chunk_size = 32768
        self.overlap_size = 0
        self.workers = 1
        self.reader = FlatReader()
        self.embed_model_name = embed_model_name

        if ('vidore' in embed_model_name or 'openbmb' in embed_model_name or 'tsystems/colqwen2.5-3b-multilingual-v1.0' in embed_model_name or 'colqwen2.5' in embed_model_name):
            if input_prefix == 'img':
                self.reader = SimpleDirectoryReader(input_dir=self.input_dir)
                # Initialize VL_Embedding for direct image processing
                self.image_embedder = VL_Embedding(model=embed_model_name, mode='image')
                # Create a simple pipeline that only handles ImageNode
                self.image_pipeline = IngestionPipeline(transformations=[])
            else:
                self.pipeline = IngestionPipeline(
                                    transformations=[
                                        SimpleFileNodeParser(),
                                        SentenceSplitter(
                                            include_metadata=True, include_prev_next_rel=True,
                                            chunk_size=self.chunk_size,
                                            chunk_overlap=self.overlap_size,
                                            separator=' ',       
                                            paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'),
                                        VL_Embedding(model=embed_model_name,mode='text')
                                    ],
                                )
        else:
            self.pipeline = IngestionPipeline(
                                transformations=[
                                    SimpleFileNodeParser(),
                                    SentenceSplitter(
                                        include_metadata=True, include_prev_next_rel=True,
                                        chunk_size=self.chunk_size,
                                        chunk_overlap=self.overlap_size,
                                        separator=' ',       
                                        paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'),
                                    HuggingFaceEmbedding(model_name=self.embed_model_name,trust_remote_code=True)
                                ],
                            )

    def create_image_node(self, image_path):
        """Create ImageNode with direct image embedding using VL_Embedding's embed_img method"""
        # Get embedding directly using embed_img
        embedding = self.image_embedder.embed_img(image_path)
        
        # Convert to List[float]
        if hasattr(embedding, 'view'):
            # Handle PyTorch tensor
            embedding = embedding.view(-1).tolist()
        elif hasattr(embedding, 'flatten'):
            # Handle numpy array
            embedding = embedding.flatten().tolist()
        elif isinstance(embedding, (list, tuple)):
            # Handle list/tuple
            embedding = [float(x) for x in embedding]
        else:
            raise ValueError(f"Unexpected embedding type: {type(embedding)}")
        
        # Create ImageNode with embedding
        image_node = ImageNode(
            image=str(image_path),  # Ensure image path is string
            metadata={
                "file_path": str(image_path),
                "file_name": os.path.basename(image_path),
                "file_type": "image"
            },
            embedding=embedding
        )
        return image_node

    def ingestion_example(self, input_file, output_file):
        # image
        if input_file.endswith('.jpg') or input_file.endswith('.png'):
            # print(f"\nProcessing image file: {input_file}")
            image_node = self.create_image_node(input_file)
            nodes = self.image_pipeline.run(documents=[image_node], show_progress=False)
        else: # txt
            documents = self.reader.load_data(Path(input_file))
            if hasattr(self, 'text_pipeline'):
                nodes = self.text_pipeline.run(documents=documents, show_progress=False)
            else:
                nodes = self.image_pipeline.run(documents=documents, show_progress=False)
        
        # Save nodes to file
        nodes_json = [node.to_dict() for node in nodes]
        with open(output_file, 'w') as json_file:
            json.dump(nodes_json, json_file, indent=2, ensure_ascii=False)
        return True

    def ingestion_multi_session(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        file_to_process = []
        for file in os.listdir(self.input_dir):
            file_prefix,_ = os.path.splitext(file)
            input_file = os.path.join(self.input_dir, file)
            output_file = os.path.join(self.output_dir, file_prefix) + '.node'
            if not os.path.exists(output_file) or os.path.getmtime(input_file) > os.path.getmtime(output_file):
                file_to_process.append((input_file, output_file))
        if self.workers == 1:
            for input_file, output_file in tqdm(file_to_process):
                self.ingestion_example(input_file, output_file)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file) for input_file, output_file in file_to_process}
                for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'):
                    result_type = future.result()
    


if __name__ == '__main__':
    root_path = './data'
    datasets = ['our_datasets']
    for dataset in datasets:
        dataset_dir = os.path.join(root_path, dataset)

        ingestion = Ingestion(dataset_dir, input_prefix='img', output_prefix='colqwen2_5_ingestion', embed_model_name='tsystems/colqwen2.5-3b-multilingual-v1.0') 
        print(f"Successfully initialized ingestion Image")
        ingestion.ingestion_multi_session()
