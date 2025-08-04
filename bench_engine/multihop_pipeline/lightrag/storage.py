import os
import json
import asyncio
from typing import Union, List, Dict, Any
import networkx as nx
from .base import BaseKVStorage, BaseGraphStorage


class JsonKVStorage(BaseKVStorage):
    def __init__(self, namespace: str, global_config: dict, embedding_func=None):
        self.namespace = namespace
        self.global_config = global_config
        self.embedding_func = embedding_func
        self.file_path = os.path.join(
            global_config["working_dir"], f"kv_store_{namespace}.json"
        )
        self._data = {}
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._data = {}

    def _save_data(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    async def filter_keys(self, keys: List[str]) -> set[str]:
        return set(k for k in keys if k not in self._data)

    async def upsert(self, data: Dict[str, Any]):
        self._data.update(data)
        self._save_data()

    async def get_by_id(self, id: str) -> Union[Dict, None]:
        return self._data.get(id)

    async def index_done_callback(self):
        self._save_data()


class NetworkXStorage(BaseGraphStorage):
    def __init__(self, namespace: str, global_config: dict, embedding_func=None):
        self.namespace = namespace
        self.global_config = global_config
        self.embedding_func = embedding_func
        self.file_path = os.path.join(
            global_config["working_dir"], f"graph_{namespace}.graphml"
        )
        self.graph = nx.Graph()
        self._load_graph()

    def _load_graph(self):
        if os.path.exists(self.file_path):
            try:

                self.graph = nx.read_graphml(self.file_path, node_type=str)

                for u, v, data in self.graph.edges(data=True):
                    if 'page' in data:
                        data['page'] = str(data['page'])
                    if 'modality' in data:
                        data['modality'] = str(data['modality'])
            except Exception as e:
                print(f"加载图文件时出错: {e}")
                self.graph = nx.Graph()
        else:
            self.graph = nx.Graph()

    def _save_graph(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        for u, v, data in self.graph.edges(data=True):
            for key, value in data.items():
                if not isinstance(value, str):
                    data[key] = str(value)

        for node, data in self.graph.nodes(data=True):
            for key, value in data.items():
                if not isinstance(value, str):
                    data[key] = str(value)

        try:
            nx.write_graphml(self.graph, self.file_path, encoding='utf-8',
                             prettyprint=True, infer_numeric_types=False)
        except Exception as e:
            print(f"保存图文件时出错: {e}")

    async def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self.graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        if self.graph.has_edge(source_node_id, target_node_id):
            return dict(self.graph.edges[source_node_id, target_node_id])
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        processed_node_data = {}
        for key, value in node_data.items():
            processed_node_data[key] = str(value) if value is not None else ""

        self.graph.add_node(node_id, **processed_node_data)
        self._save_graph()

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        processed_edge_data = {}
        for key, value in edge_data.items():
            processed_edge_data[key] = str(value) if value is not None else ""

        if 'page' not in processed_edge_data:
            processed_edge_data['page'] = "unknown"
        if 'modality' not in processed_edge_data:
            processed_edge_data['modality'] = "unknown"

        self.graph.add_edge(source_node_id, target_node_id, **processed_edge_data)
        self._save_graph()

    async def index_done_callback(self):
        self._save_graph()