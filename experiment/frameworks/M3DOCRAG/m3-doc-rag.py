import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    AutoConfig
)
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from pdf2image import convert_from_path
from qwen_vl_utils import process_vision_info
from PIL import Image
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import cmd
import os
import traceback
from openai import OpenAI


@dataclass
class DocumentPage:
    """Represents a single page from a document with its metadata."""
    doc_id: str
    page_num: int
    image: Image.Image  

@dataclass
class RetrievedPage:
    """Represents a retrieved page with its relevance score."""
    page: DocumentPage
    score: float

class M3DOCRAG:
    def __init__(
        self,
        openai_api_key: str,
        retrieval_model_name: str = "vidore/colpali",
        colpali_base_name: str = "vidore/colpaligemma-3b-mix-448-base",
        max_pages: int = 4,
        use_approximate_index: bool = True,
        batch_size: int = 4,
    ):
        """Initialize M3DOCRAG framework with OpenAI for QA."""
        self.max_pages = max_pages
        self.use_approximate_index = use_approximate_index
        self.batch_size = batch_size
        
        # Setup OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Setup device for retrieval model
        self.retrieval_device = "cuda:0"
        print(f"Using retrieval device: {self.retrieval_device}")

        print("Initializing retrieval model...")
        self.retrieval_model = ColPali.from_pretrained(
            colpali_base_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self.retrieval_device}
        ).eval()
        self.retrieval_model.load_adapter(retrieval_model_name)
        self.retrieval_processor = AutoProcessor.from_pretrained(retrieval_model_name)
        
        self.retriever_evaluator = CustomEvaluator(is_multi_vector=True)
        self.index = None
        self.pages: List[DocumentPage] = []
        print("Models initialized successfully!")

    def add_document(self, pdf_path: str, doc_id: str):
        """Add a PDF document to the corpus."""
        print(f"Loading document: {doc_id}")
        page_images = convert_from_path(pdf_path, dpi=144)
        for page_num, image in enumerate(page_images):
            page = DocumentPage(
                doc_id=doc_id,
                page_num=page_num,
                image=image
            )
            self.pages.append(page)
        print(f"Added {len(page_images)} pages from {doc_id}")
    
    def encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def build_index(self):
        """Build the retrieval index for all pages."""
        print(f"Building index for {len(self.pages)} pages...")
        
        total_batches = (len(self.pages) + self.batch_size - 1) // self.batch_size
        
        dataloader = DataLoader(
            self.pages,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: process_images(self.retrieval_processor, [page.image for page in x])
        )

        all_embeddings = []
        try:
            for batch_doc in tqdm(dataloader, desc="Building index", total=total_batches):
                with torch.no_grad():
                    batch_doc = {
                        k: v.to(self.retrieval_device, dtype=torch.bfloat16) 
                        if k == "pixel_values" else v.to(self.retrieval_device)
                        for k, v in batch_doc.items()
                    }
                    
                    with torch.cuda.device(self.retrieval_device):
                        torch.cuda.empty_cache()
                    
                    embeddings_doc = self.retrieval_model(**batch_doc)
                    embeddings_doc = embeddings_doc.to(dtype=torch.float32)
                    embeddings_doc = embeddings_doc.mean(dim=1)
                    
                    embeddings_doc = embeddings_doc.cpu()
                    all_embeddings.extend(list(torch.unbind(embeddings_doc)))

            all_embeddings = np.stack([emb.numpy() for emb in all_embeddings])
            print(f"Embeddings shape: {all_embeddings.shape}")
            embedding_dim = all_embeddings.shape[1]
            n_vectors = all_embeddings.shape[0]

            if self.use_approximate_index and n_vectors >= 156:
                print("Building approximate index...")
                quantizer = faiss.IndexFlatIP(embedding_dim)
                n_centroids = max(1, min(
                    n_vectors // 40,
                    int(np.sqrt(n_vectors)),
                    100
                ))
                print(f"Using {n_centroids} centroids for IVF index")
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    embedding_dim,
                    n_centroids,
                    faiss.METRIC_INNER_PRODUCT
                )
                self.index.train(all_embeddings)
            else:
                print("Building exact index...")
                self.index = faiss.IndexFlatIP(embedding_dim)

            self.index.add(all_embeddings)
            print("Index built successfully!")
            
        except Exception as e:
            print(f"Error during index building: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            raise

    def retrieve(self, query: str) -> List[RetrievedPage]:
        """Retrieve relevant pages."""
        try:
            with torch.cuda.device(self.retrieval_device):
                torch.cuda.empty_cache()

            dummy_image = Image.new("RGB", (448, 448), (255, 255, 255))
            query_batch = self.process_images(
                self.retrieval_processor,
                [dummy_image]
            )

            with torch.no_grad():
                query_batch = {
                    k: v.to(self.retrieval_device, dtype=torch.bfloat16)
                    if k == "pixel_values" else v.to(self.retrieval_device)
                    for k, v in query_batch.items()
                }
                query_embedding = self.retrieval_model(**query_batch)
                query_embedding = query_embedding.to(dtype=torch.float32)
                query_embedding = query_embedding.mean(dim=1)
                query_embedding_np = query_embedding.cpu().numpy()

            scores, indices = self.index.search(query_embedding_np, self.max_pages)

            retrieved_pages = []
            for score, idx in zip(scores[0], indices[0]):
                retrieved_pages.append(
                    RetrievedPage(
                        page=self.pages[idx],
                        score=float(score)
                    )
                )

            return retrieved_pages

        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return []

    def process_images(self, processor, pages):
        """Process images with proper token handling."""
        n_images = len(pages)
        text = "<image>" * n_images + "<bos>"
        
        return processor(
            text=text,
            images=pages,
            return_tensors="pt"
        )
    
    def answer(self, query: str, retrieved_pages: List[RetrievedPage]) -> str:
        """Generate answer using OpenAI's GPT-4 Vision."""
        try:
            # Prepare the messages
            messages = [
                {
                    "role": "system",
                    "content": """You are a document analyzer that ONLY gives two types of responses:
                    1. If you find the EXACT information: Respond with ONLY that specific information
                    2. If you cannot find the EXACT information: Respond with EXACTLY and ONLY this phrase: "I cannot find this information in the provided document pages."
                    3. Answer in the same of the query language.

                    DO NOT:
                    - Explain your limitations
                    - Talk about AI or models
                    - Make assumptions
                    - Give partial information
                    - Provide multiple answers
                    - Add any explanations"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"IMPORTANT: Give ONLY the exact answer found in these document pages for: {query}"
                        },
                        *[{
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.encode_image(page.page.image)}"
                            }
                        } for page in retrieved_pages]
                    ]
                }
            ]

            # Get response from OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=50,
                temperature=0.1,
            )

            raw_answer = response.choices[0].message.content.strip()
            
            # Process the answer to ensure compliance
            return raw_answer

        except Exception as e:
            print(f"Error in answer generation: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return f"Error: {str(e)}"

    def process_query(self, query: str) -> str:
        """Process a query and return the answer."""
        print("\nProcessing query:", query)
        print("Retrieving relevant pages...")
        retrieved_pages = self.retrieve(query)
        print(f"Retrieved {len(retrieved_pages)} pages")
        print("Generating answer...")
        answer = self.answer(query, retrieved_pages)
        return answer

class M3DOCRAGShell(cmd.Cmd):
    intro = 'Welcome to M3DOCRAG interactive shell. Type help or ? to list commands.\n'
    prompt = '(M3DOCRAG) '

    def __init__(self):
        super().__init__()
        self.rag = None
        self.documents = {}

    def do_init(self, arg):
        """Initialize the M3DOCRAG system: init <openai_api_key>"""
        if not arg:
            print("Please provide your OpenAI API key")
            return
            
        print("Initializing M3DOCRAG system...")
        try:
            self.rag = M3DOCRAG(openai_api_key=arg.strip())
            print("System initialized!")
        except Exception as e:
            print(f"Error initializing system: {e}")

    def do_add(self, arg):
        """Add a PDF document: add <pdf_path>"""
        if not self.rag:
            print("Please initialize the system first using 'init'")
            return
        
        try:
            pdf_path = arg.strip()
            if not os.path.exists(pdf_path):
                print(f"Error: File not found: {pdf_path}")
                return
                
            doc_id = os.path.basename(pdf_path)
            self.documents[doc_id] = pdf_path
            self.rag.add_document(pdf_path, doc_id)
            print(f"Added document: {doc_id}")
        except Exception as e:
            print(f"Error adding document: {e}")

    def do_build(self, arg):
        """Build the index after adding documents"""
        if not self.rag:
            print("Please initialize the system first using 'init'")
            return
        
        if not self.rag.pages:
            print("Please add documents first using 'add'")
            return
            
        try:
            self.rag.build_index()
        except Exception as e:
            print(f"Error building index: {e}")

    def do_ask(self, arg):
        """Ask a question: ask <question>"""
        if not self.rag or not self.rag.index:
            print("Please initialize the system and build the index first")
            return
            
        try:
            answer = self.rag.process_query(arg)
            print("\nQuestion:", arg)
            print("Answer:", answer)
        except Exception as e:
            print(f"Error processing question: {e}")

    def do_list(self, arg):
        """List all loaded documents"""
        if not self.documents:
            print("No documents loaded")
            return
            
        print("\nLoaded documents:")
        for doc_id, path in self.documents.items():
            print(f"- {doc_id}: {path}")

    def do_exit(self, arg):
        """Exit the program"""
        print("Goodbye!")
        return True

def main():
    shell = M3DOCRAGShell()
    shell.cmdloop()

if __name__ == "__main__":
    main()