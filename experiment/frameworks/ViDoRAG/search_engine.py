from typing import Optional, List, Mapping, Any, Dict
import json
from tqdm import tqdm
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.mixture import GaussianMixture
import datetime

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.schema import NodeWithScore, BaseNode, MetadataMode, IndexNode, ImageNode, TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llms.vl_embedding import VL_Embedding
from utils.format_converter import nodefile2node, nodes2dict


def gmm(recall_result: list[NodeWithScore], input_length: int = 20, max_valid_length: int = 10, min_valid_length: int = 5) -> List[NodeWithScore]:
    scores = [node.score for node in recall_result[:input_length]]
    scores = np.array(scores)
    scores = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=1, random_state=0)
    gmm.fit(scores)
    labels = gmm.predict(scores)

    scores = scores.flatten()
    scores = [scores[labels == label] for label in np.unique(labels)]
    recall_result = [np.array(recall_result[:input_length])[labels == label].tolist() for label in np.unique(labels)]

    max_values = np.array([np.max(p) for p in scores])
    sorted_indices = np.argsort(-max_values)

    if len(sorted_indices) == 1:
        valid_recall_result = recall_result[0]
        valid_recall_result = valid_recall_result[:max_valid_length]
        for node in valid_recall_result:
            node.score = None
        return valid_recall_result

    max_index = sorted_indices[0]
    second_max_index = sorted_indices[1]

    valid_recall_result = recall_result[max_index]

    if len(valid_recall_result) > max_valid_length:
        valid_recall_result = valid_recall_result[:max_valid_length]
    elif len(valid_recall_result) < min_valid_length:
        second_valid_recall_result_len = min_valid_length - len(valid_recall_result)
        valid_recall_result.extend(recall_result[second_max_index][:second_valid_recall_result_len])

    for node in valid_recall_result:
        node.score = None

    return valid_recall_result

class SearchEngine:
    def __init__(self, dataset, node_dir_prefix=None, embed_model_name='BAAI/bge-m3'):
        Settings.llm = None
        self.gmm = False
        self.gmm_candidate_length = False
        self.return_raw = False
        self.input_gmm = 20
        self.max_output_gmm = 10
        self.min_output_gmm = 5
        self.dataset = dataset
        self.dataset_dir = os.path.join('path/to/your/data', dataset)
        self.search_history = []
        if node_dir_prefix is None:
            if 'bge' in embed_model_name:
                node_dir_prefix = 'bge_ingestion'
            elif 'NV-Embed' in embed_model_name:
                node_dir_prefix = 'nv_ingestion'
            elif 'colqwen' in embed_model_name:
                if 'colqwen2_5' in embed_model_name or 'colqwen2.5' in embed_model_name:
                    node_dir_prefix = 'colqwen2_5_ingestion'
                else:
                    node_dir_prefix = 'colqwen_ingestion'
            elif 'openbmb' in embed_model_name:
                node_dir_prefix = 'visrag_ingestion'
            elif 'colpali' in embed_model_name:
                node_dir_prefix = 'colpali_ingestion'
            else:
                raise ValueError('Please specify the node_dir_prefix')
        if node_dir_prefix in ['colqwen_ingestion', 'visrag_ingestion', 'colpali_ingestion', 'colqwen2_5_ingestion']:
            self.vl_ret = True
        else:
            self.vl_ret = False

        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.rag_dataset_path = os.path.join(self.dataset_dir, 'rag_dataset.json')
        self.workers = 1
        self.embed_model_name = embed_model_name
        print(f'self.vl_ret: {self.vl_ret}')
        print(f'self.embed_model_name: {self.embed_model_name}')
        if 'colqwen2.5' in embed_model_name or 'openbmb' in embed_model_name or 'colqwen' in embed_model_name:
            if self.vl_ret:
                self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')
            else:
                self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='text')
        else:
            self.vector_embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, embed_batch_size=10, max_length=512, trust_remote_code=True, device='cuda')
        self.recall_num = 100
        self.query_engine = self.load_query_engine()
        self.output_dir = os.path.join(self.dataset_dir, 'search_output')

    def online_search(self, query, node_list, topk=9):
        nodes = [TextNode.from_dict(node['node']) for node in node_list]
        vector_index = VectorStoreIndex(nodes, embed_model=self.vector_embed_model, show_progress=True, use_async=False, insert_batch_size=2048)
        vector_retriever = vector_index.as_retriever(similarity_top_k=topk)
        node_postprocessors = self.load_node_postprocessors()
        query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            node_postprocessors=node_postprocessors
        )
        query_bundle = QueryBundle(query_str=query)
        recall_results = query_engine.retrieve(query_bundle)
        return nodes2dict(recall_results)

    def load_nodes(self):
        files = os.listdir(self.node_dir)
        parsed_files = []
        max_workers = 10
        if max_workers == 1:
            for file in tqdm(files):
                input_file = os.path.join(self.node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    continue
                nodes = nodefile2node(input_file)
                parsed_files.extend(nodes)
        else:
            def parse_file(file, node_dir):
                input_file = os.path.join(node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    return []
                return nodefile2node(input_file)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(parse_file, files, [self.node_dir]*len(files)), total=len(files)))
            for result in results:
                parsed_files.extend(result)
        return parsed_files

    def load_query_engine(self):
        print('Loading nodes...')
        self.nodes = self.load_nodes()

        if 'colqwen2.5' in self.embed_model_name:
            self.embeddings = [torch.tensor(node.embedding).view(-1,128).bfloat16()
                             for node in self.nodes]
            self.embeddings = [tensor.to(self.vector_embed_model.embed_model.device)
                             for tensor in self.embeddings]
            if self.vl_ret:
                self.embedding_img = self.embeddings
            else:
                self.embedding_text = self.embeddings
        else:
            retriever = self.load_retriever_embed(self.nodes)
            node_postprocessors = self.load_node_postprocessors()
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=node_postprocessors
            )
            return query_engine

    def load_node_postprocessors(self):
        return []

    def load_retriever_embed(self, nodes):
        for node in nodes:
            if hasattr(node, 'embedding') and node.embedding is not None:
                if isinstance(node.embedding, torch.Tensor):
                    node.embedding = node.embedding.float().cpu().numpy()
                elif isinstance(node.embedding, np.ndarray):
                    node.embedding = node.embedding.astype(np.float32)

        vector_index = VectorStoreIndex(nodes, embed_model=self.vector_embed_model, show_progress=True, use_async=False, insert_batch_size=2048)
        vector_retriever = vector_index.as_retriever(similarity_top_k=self.recall_num)
        return vector_retriever

    def record_search_result(self, query, recall_result):
        search_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'results': recall_result,
            'num_results': len(recall_result['source_nodes']) if recall_result and 'source_nodes' in recall_result else 0
        }
        self.search_history.append(search_record)
        return search_record

    def get_search_history(self):
        return self.search_history

    def save_search_history(self, output_file='search_history.json'):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, output_file), 'w') as f:
            json.dump(self.search_history, f, indent=2, ensure_ascii=False)

    def search(self, query):
        query_embedding = self.vector_embed_model.embed_text(query)

        if 'colqwen2.5' in self.embed_model_name:
            if self.vl_ret:
                scores = self.vector_embed_model.processor.score(query_embedding, self.embedding_img)
            else:
                scores = self.vector_embed_model.processor.score(query_embedding, self.embedding_text)
        else:
            scores = self.vector_index.similarity_search_with_score(query_embedding)
            scores = torch.tensor([score for _, score in scores])

        k = min(100, scores[0].numel())
        values, indices = torch.topk(scores[0], k=k)
        recall_results = [self.nodes[i] for i in indices]

        for node in recall_results:
            node.embedding = None

        recall_results = [NodeWithScore(node=node, score=float(score.item()))
                         for node, score in zip(recall_results, values)]
        recall_results_output = recall_results

        if self.gmm:
            recall_results_output = gmm(recall_results, self.input_gmm, self.max_output_gmm, self.min_output_gmm)
        if self.return_raw:
            return recall_results_output
        if self.gmm_candidate_length:
            candidate_length = [1, 2, 4, 6, 9, 12, 16, 20]
            current_length = len(recall_results_output)
            target_length = min([num for num in candidate_length if num > current_length])
            recall_results_output = recall_results[:target_length]
        result = nodes2dict(recall_results_output)
        self.record_search_result(query, result)
        return result

    def search_example(self, example):
        query = example['query']
        recall_result = self.search(query)
        example['recall_result'] = recall_result
        return example

    def search_multi_session(self, output_file='search_result.json'):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.rag_dataset_path, 'r') as f:
            dataset = json.load(f)
        data = dataset['examples']
        results = []
        if self.workers == 1:
            for example in tqdm(data):
                results.append(self.search_example(example))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.search_example, example): example for example in data}
                for future in tqdm(as_completed(future_to_file), total=len(data), desc='Processing files'):
                    results.append(future.result())
        with open(os.path.join(self.output_dir, output_file), 'w') as json_file:
            json.dump(results, json_file, indent=2, ensure_ascii=False)

class HybridSearchEngine:
    def __init__(
        self,
        dataset,
        rag_dataset_file=None,
        node_dir_prefix_vl=None,
        node_dir_prefix_text=None,
        embed_model_name_vl='tsystems/colqwen2.5-3b-multilingual-v1.0',
        embed_model_name_text='tsystems/colqwen2.5-3b-multilingual-v1.0',
        topk=10,
        gmm=False
    ):
        self.dataset = dataset
        self.dataset_dir = os.path.join('path/to/your/data', dataset)
        self.img_dir = os.path.join(self.dataset_dir, 'img')
        self.ppocr_dir = os.path.join(self.dataset_dir, 'txt')
        self.engine_vl = SearchEngine(dataset, node_dir_prefix=node_dir_prefix_vl, embed_model_name=embed_model_name_vl)
        self.engine_text = SearchEngine(dataset, node_dir_prefix=node_dir_prefix_text, embed_model_name=embed_model_name_text)
        self.topk = topk
        self.gmm = gmm
        self.rag_dataset_path = os.path.join(self.dataset_dir, rag_dataset_file)
        self.workers = 10
        self.search_history = []
        self.output_dir = os.path.join(self.dataset_dir, 'hybrid_search_output')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f'self.rag_dataset_path: {self.rag_dataset_path}')

    def record_search_result(self, query, recall_result):
        search_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'results': recall_result,
            'num_results': len(recall_result['source_nodes']) if recall_result and 'source_nodes' in recall_result else 0
        }
        self.search_history.append(search_record)
        return search_record

    def get_search_history(self):
        return self.search_history

    def save_search_history(self, output_file='hybrid_search_history.json'):
        output_dir = os.path.join(self.dataset_dir, 'hybrid_search_output')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, output_file), 'w') as f:
            json.dump(self.search_history, f, indent=2, ensure_ascii=False)

    def search(self, query):
        union_result = False
        if union_result:
            if self.gmm:
                self.engine_vl.gmm = True
                self.engine_text.gmm = True
                self.engine_vl.input_gmm = self.topk * 2
                self.engine_text.input_gmm = self.topk * 2
                self.engine_vl.max_output_gmm = self.topk
                self.engine_text.max_output_gmm = self.topk
                self.engine_vl.min_output_gmm = self.topk // 2
                self.engine_text.min_output_gmm = self.topk // 2
            result_vl = self.engine_vl.search(query)
            result_text = self.engine_text.search(query)
            result_vl['source_nodes'] = result_vl['source_nodes'][:self.topk]
            result_text['source_nodes'] = result_text['source_nodes'][:self.topk]

            result_docs = dict()
            for node in result_vl['source_nodes']:
                file = os.path.basename(node['node']['image_path']).split('.')[0]
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))

            for node in result_text['source_nodes']:
                file = node['node']['metadata']['filename'].split('.')[0]
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))

            recall_result = []

            for key, pages in result_docs.items():
                pages = sorted(pages)
                for page in pages:
                    with open(os.path.join(self.ppocr_dir, f'{key}_{page}.txt'), 'r') as f:
                        text = f.readlines()
                    text = ' '.join([item.strip() for item in text])
                    file_path = os.path.join(self.img_dir, f'{key}_{page}.jpg')
                    node = ImageNode(image_path=file_path, text=text, metadata=dict(file_name=file_path))
                    recall_result.append(NodeWithScore(node=node, score=None))

            result = nodes2dict(recall_result)
            self.record_search_result(query, result)
            return result

        else:
            self.engine_vl.return_raw = True
            self.engine_text.return_raw = True

            result_vl = self.engine_vl.search(query)
            result_text = self.engine_text.search(query)

            result_vl_gmm = gmm(result_vl, self.topk * 2, self.topk, 5)
            result_text_gmm = gmm(result_text, self.topk * 2, self.topk, 5)

            result_vl_gmm = nodes2dict(result_vl_gmm)
            result_text_gmm = nodes2dict(result_text_gmm)

            result_docs = dict()
            for node in result_vl_gmm['source_nodes']:
                image_path = node['node'].get('image_path')
                if not image_path or not isinstance(image_path, str):
                    continue
                file = '.'.join(os.path.basename(image_path).split('.')[:-1])
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))
            for node in result_text_gmm['source_nodes']:
                file = '.'.join(node['node']['metadata']['filename'].split('.')[:-1])
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))

            result_docs_list = []
            for key, pages in result_docs.items():
                for page in pages:
                    result_docs_list.append(f'{key}_{page}')

            result_vl = nodes2dict(result_vl)
            result_text = nodes2dict(result_text)
            result_docs_text_list = []
            result_docs_vl_list = []
            for node in result_vl['source_nodes']:
                image_path = node['node'].get('image_path')
                if not image_path or not isinstance(image_path, str):
                    continue
                file = '.'.join(os.path.basename(image_path).split('.')[:-1])
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                result_docs_vl_list.append(doc + '_' + page)
            for node in result_text['source_nodes']:
                file = '.'.join(node['node']['metadata']['filename'].split('.')[:-1])
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                result_docs_text_list.append(doc + '_' + page)

            overleap = [doc for doc in result_docs_vl_list if doc in result_docs_text_list]

            target_length = 5
            already_length = sum([len(value) for _, value in result_docs.items()])

            if already_length > target_length:
                for key in list(result_docs.keys()):
                    result_docs[key] = sorted(result_docs[key])[:target_length]
                    if not result_docs[key]:
                        del result_docs[key]
                already_length = target_length

            candidate_overleap = [node for node in overleap if node not in result_docs_list][:target_length - already_length]

            for file in candidate_overleap:
                doc = '_'.join(file.split('_')[:-1])
                page = file.split('_')[-1]
                if doc not in result_docs:
                    result_docs[doc] = [int(page)]
                else:
                    if int(page) not in result_docs[doc]:
                        result_docs[doc].append(int(page))

            recall_result = []
            for key, pages in result_docs.items():
                pages = sorted(pages)
                for page in pages:
                    with open(os.path.join(self.ppocr_dir, f'{key}_{page}.txt'), 'r') as f:
                        text = f.readlines()
                    text = ' '.join([item.strip() for item in text])
                    text_path = os.path.join(self.ppocr_dir, f'{key}_{page}.txt')
                    image_path = os.path.join(self.img_dir, f'{key}_{page}.jpg')
                    node = ImageNode(
                        image_path=image_path,
                        text_path=text_path,
                        text=text,
                        metadata=dict(text_path=text_path, image_path=image_path)
                    )
                    recall_result.append(NodeWithScore(node=node, score=None))

            result = nodes2dict(recall_result)
            self.record_search_result(query, result)
            return result

    def search_example(self, example):
        query = example['question']
        recall_result = self.search(query)
        example['recall_result'] = recall_result
        return example

    def search_multi_session(self, output_file='search_result.json'):
        base_name = os.path.splitext(os.path.basename(self.rag_dataset_path))[0]
        output_file = f"{base_name}_search_result.json"
        output_path = os.path.join(self.output_dir, output_file)

        processed_results = []
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                processed_results = json.load(f)

        processed_queries = {result['question'] for result in processed_results}

        with open(self.rag_dataset_path, 'r') as f:
            data = json.load(f)

        data = [example for example in data if example['question'] not in processed_queries]

        if not data:
            print("All examples have been processed.")
            return

        if self.workers == 1:
            for example in tqdm(data, desc="Processing examples"):
                try:
                    result = self.search_example(example)
                    processed_results.append(result)
                    with open(output_path, 'w') as json_file:
                        json.dump(processed_results, json_file, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error processing example: {example['question']}")
                    print(f"Error details: {str(e)}")
                    with open(output_path, 'w') as json_file:
                        json.dump(processed_results, json_file, indent=2, ensure_ascii=False)
                    raise e
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_example = {executor.submit(self.search_example, example): example for example in data}
                for future in tqdm(as_completed(future_to_example), total=len(data), desc='Processing files'):
                    try:
                        result = future.result()
                        processed_results.append(result)
                        with open(output_path, 'w') as json_file:
                            json.dump(processed_results, json_file, indent=2, ensure_ascii=False)
                    except Exception as e:
                        example = future_to_example[future]
                        print(f"Error processing example: {example['question']}")
                        print(f"Error details: {str(e)}")
                        with open(output_path, 'w') as json_file:
                            json.dump(processed_results, json_file, indent=2, ensure_ascii=False)
                        raise e

if __name__ == '__main__':
    datasets = ['your_dataset']
    for dataset in datasets:
        search_engine = HybridSearchEngine(
            dataset,
            node_dir_prefix_vl='colqwen2_5_ingestion',
            node_dir_prefix_text='colqwen2_5_ingestion_text',
            rag_dataset_file='path/to/your/benchmark.json',
            embed_model_name_vl='tsystems/colqwen2.5-3b-multilingual-v1.0',
            embed_model_name_text='tsystems/colqwen2.5-3b-multilingual-v1.0',
            gmm=True
        )
        result = search_engine.search_multi_session()
        search_engine.save_search_history()