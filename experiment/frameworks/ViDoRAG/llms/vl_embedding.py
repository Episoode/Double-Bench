import asyncio
from typing import Any, List, Optional, Union
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.embeddings.base import Embedding

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor, ColPali, ColPaliProcessor


def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

class VL_Embedding(MultiModalEmbedding):

    model: str = Field(description="The Multi-model to use.")

    api_key: Optional[str] = Field(
        default=None,
        description="The API key.",
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description=(
            "The number of dimensions the resulting output embeddings should have. "
            "Only supported in embedding-3 and later models. embedding-2 is fixed at 1024."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description="The timeout.",
    )

    mode: str = Field(
        default='text',
        description="The mode of the model, either 'text' or 'image'."
    )
    show_progress: bool = Field(
        default=False,
        description="Whether to show progress bars.",
    )
    
    embed_model: Union[ColQwen2_5, AutoModel, None] = Field(
        default=None
    )
    processor: Optional[ColQwen2_5_Processor] = Field(
        default=None
    )
    tokenizer: Optional[AutoTokenizer] = Field(
        default=None
    )
    
    
    def __init__(
        self,
        model: str = "tsystems/colqwen2.5-3b-multilingual-v1.0",
        dimensions: Optional[int] = 1024,
        timeout: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        mode: str = 'text',
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )
        
        self.mode = mode
        
        if 'openbmb' in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self.embed_model = AutoModel.from_pretrained(model,
             torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto').eval()
            # self.embed_model.eval()
        elif ('colqwen2.5' in model or 'colqwen2_5' in model or 'colqwen2-5' in model or 'tsystems/colqwen2.5-3b-multilingual-v1.0' in model or model.startswith('models--tsystems--colqwen2.5-3b-multilingual-v1.0')):
            hf_model_name = "tsystems/colqwen2.5-3b-multilingual-v1.0"
            self.embed_model = ColQwen2_5.from_pretrained(
                hf_model_name,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                cache_dir='/path/to/your/cache/'
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(hf_model_name, use_fast=True, cache_dir='')
        elif 'vidore' in model and 'pali' in model:
            self.embed_model = ColPali.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model)
        
        


    @classmethod
    def class_name(cls) -> str:
        return "VL_Embedding"
    
    def embed_img(self, img_path):
        if isinstance(img_path, str):
            img_path = [img_path]
        # print(f"Processing: {img_path}")
        hf_model_name = "tsystems/colqwen2.5-3b-multilingual-v1.0"
        if (self.model == hf_model_name or self.model.startswith('models--tsystems--colqwen2.5-3b-multilingual-v1.0')):
            images = [Image.open(img) for img in img_path]
            batch_images = self.processor.process_images(images).to(self.embed_model.device)
            with torch.no_grad():
                image_embeddings = self.embed_model(**batch_images)
        elif 'openbmb' in self.model:
            images = [Image.open(img).convert('RGB') for img in img_path]
            inputs = {
                "text": [''] * len(images),
                'image': images,
                'tokenizer': self.tokenizer
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)   
                image_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                # image_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().tolist()[0]
            # image_embeddings = embeddings.tolist()[0]
        return image_embeddings
    
    def embed_text(self, text):
        if isinstance(text, str):
            text = [text]
        hf_model_name = "tsystems/colqwen2.5-3b-multilingual-v1.0"
        if (self.model == hf_model_name or self.model.startswith('models--tsystems--colqwen2.5-3b-multilingual-v1.0')):
            batch_queries = self.processor.process_queries(text).to(self.embed_model.device)
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif 'colpali' in self.model:
            batch_queries = self.processor.process_queries(text).to(self.embed_model.device)
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif 'openbmb' in self.model:
            INSTRUCTION = "Represent this query for retrieving relevant documents: "
            queries = [INSTRUCTION + query for query in text]
            inputs = {
                "text": queries,
                'image': [None] * len(queries),
                'tokenizer': self.tokenizer
                }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)   
                # query_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                query_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().tolist()
                # query_embeddings = embeddings.tolist()[0]
        return query_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.embed_text(text)
            embeddings = embeddings[0]
            embeddings_list.append(embeddings)
        return embeddings_list

    def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]
    
    def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]
    
    def _get_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)
    
    def _aget_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)
    
    def __call__(self, nodes, **kwargs):
        hf_model_name = "tsystems/colqwen2.5-3b-multilingual-v1.0"
        if (self.model == hf_model_name or self.model.startswith('models--tsystems--colqwen2.5-3b-multilingual-v1.0')):
            if self.mode == 'image':
                embeddings = self.embed_img([node.metadata['file_path'] for node in nodes])
                embeddings = embeddings.view(embeddings.size(0),-1).tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                embeddings = embeddings.view(embeddings.size(0),-1).tolist()

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
                
        elif 'openbmb' in self.model:
            if self.mode == 'image':
                embeddings = self.embed_img([node.metadata['file_path'] for node in nodes])
                embeddings = embeddings.tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                # embeddings = embeddings.tolist()
                # embeddings = [embeddings]

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
                
        return nodes
    
    def score(self,image_embeddings,text_embeddings):
        hf_model_name = "tsystems/colqwen2.5-3b-multilingual-v1.0"
        if (self.model == hf_model_name or self.model.startswith('models--tsystems--colqwen2.5-3b-multilingual-v1.0')):
            score = self.processor.score_multi_vector(image_embeddings, text_embeddings)
        elif 'openbmb' in self.model:
            score = text_embeddings @ image_embeddings.T
        return score

if __name__ == "__main__":
    import json
    colqwen2_5_path = "models--tsystems--colqwen2.5-3b-multilingual-v1.0"
    colqwen2_5 = VL_Embedding(colqwen2_5_path)
    image_embeddings = colqwen2_5.embed_img("/mnt/nvme1/wmj/frameworks/ViDoRAG/data/our_datasets/img/0b85477387a9d0cc33fca0f4becaa0e5_0.jpg")
    print("Successfully embed image")
    # save embeddings with json format
    with open("embeddings.json", "w") as f:
        json.dump(image_embeddings, f)
    # score = colqwen2_5.processor.score_multi_vector(image_embeddings, text_embeddings)
    # print("Successfully score")
