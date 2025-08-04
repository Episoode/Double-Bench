import os
import asyncio
from tqdm import tqdm
from lightrag import LightRAG
from lightrag.llm import gpt_4o_complete
import warnings
from bench_engine.config import OPENAI_API_KEY, TARGET_DIR

warnings.filterwarnings("ignore")


class KnowledgeGraphBuilder:
    """
    Knowledge graph builder class for constructing knowledge graphs from text files.

    This class provides functionality to build knowledge graphs using LightRAG from
    various types of text files including pure text, table text, and figure text.
    """

    def __init__(self, working_dir=None, **rag_kwargs):
        """
        Initialize the knowledge graph builder.

        Args:
            working_dir: Working directory path for storing graph data
            **rag_kwargs: Additional configuration parameters for LightRAG
        """
        self.working_dir = working_dir
        self.rag = None

        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

        self.default_rag_config = {
            "llm_model_func": gpt_4o_complete,
            "llm_model_max_async": 16,
            "llm_model_max_token_size": 32768,
            "llm_model_kwargs": {"temperature": 0.0},
            "kv_storage": "JsonKVStorage",
            "graph_storage": "NetworkXStorage",
            "chunk_token_size": 500,
            "chunk_overlap_token_size": 100,
            "entity_summary_to_max_tokens": 500,
        }

        self.rag_config = {**self.default_rag_config, **rag_kwargs}

    def init_rag(self, working_dir=None):
        """
        Initialize LightRAG instance with specified configuration.

        Args:
            working_dir: Working directory path. If None, uses the path set during initialization

        Returns:
            LightRAG: Initialized LightRAG instance

        Raises:
            ValueError: If working directory is not specified
        """
        if working_dir:
            self.working_dir = working_dir

        if not self.working_dir:
            raise ValueError("Working directory must be specified")

        self.rag = LightRAG(
            working_dir=self.working_dir,
            **self.rag_config
        )
        return self.rag

    @staticmethod
    def remove_pipes(input_string):
        """
        Remove pipe symbols from input string between first and last occurrence.

        Args:
            input_string: Input string to process

        Returns:
            str: Processed string with content between first and last pipes removed
        """
        first_pipe_index = input_string.find('|')
        last_pipe_index = input_string.rfind('|')
        if first_pipe_index == -1 or first_pipe_index == last_pipe_index:
            return input_string
        before_first_pipe = input_string[:first_pipe_index]
        after_last_pipe = input_string[last_pipe_index + 1:]
        return before_first_pipe + after_last_pipe

    @staticmethod
    def get_file_metadata(file_path, file_type):
        """
        Extract page and modality information from file path based on file type.

        Args:
            file_path: Path to the file
            file_type: Type of file (pure_text, table_text, figure_text)

        Returns:
            tuple: (page, modality) extracted from filename
        """
        basename = os.path.basename(file_path)

        if file_type == "pure_text":
            page = basename.split('.txt')[0]
            modality = "text"
        elif file_type == "table_text":
            page = basename.split('_')[0]
            modality = "table"
        elif file_type == "figure_text":
            page = basename.split('_')[0]
            modality = "figure"
        else:
            page = "unknown"
            modality = "unknown"

        return page, modality

    async def insert_texts(self, text_files_with_metadata, max_concurrent=16):
        """
        Batch insert text files into the knowledge graph with metadata.

        Args:
            text_files_with_metadata: List of file information tuples (file_path, file_type)
            max_concurrent: Maximum number of concurrent insertions

        Raises:
            ValueError: If RAG instance is not initialized
        """
        if not self.rag:
            raise ValueError("RAG instance not initialized. Please call init_rag() method first")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def insert_one_text(file_info):
            async with semaphore:
                text_path, file_type = file_info
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = self.remove_pipes(content)

                page, modality = self.get_file_metadata(text_path, file_type)
                await self.rag.ainsert(content, file_metadata={
                    "page": page,
                    "modality": modality,
                    "file_path": text_path
                })

        tasks = [insert_one_text(file_info) for file_info in text_files_with_metadata]
        await asyncio.gather(*tasks)

    def collect_text_files(self, base_dir):
        """
        Collect all text files from specified directory structure.

        Args:
            base_dir: Base directory path containing text subdirectories

        Returns:
            list: List of file information tuples (file_path, file_type)
        """
        text_files_with_metadata = []

        text_dirs = {
            "pure_text": os.path.join(base_dir, 'text'),
            "table_text": os.path.join(base_dir, 'table_text'),
            "figure_text": os.path.join(base_dir, 'figure_text')
        }

        for file_type, dir_path in text_dirs.items():
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)

                if file_type in ["table_text", "figure_text"]:
                    keyword = file_type.split('_')[0]
                    files = [f for f in files if keyword not in f]

                file_paths = [os.path.join(dir_path, file) for file in files]
                text_files_with_metadata.extend([(path, file_type) for path in file_paths])

        return text_files_with_metadata

    async def build_knowledge_graph_for_directory(self, directory_path):
        """
        Build knowledge graph for a single directory.

        Args:
            directory_path: Path to the directory containing text files
        """
        wkdir = os.path.join(directory_path, 'wkdir')
        os.makedirs(wkdir, exist_ok=True)

        self.init_rag(wkdir)

        text_files_with_metadata = self.collect_text_files(directory_path)

        if text_files_with_metadata:
            await self.insert_texts(text_files_with_metadata)

    async def build_knowledge_graphs_batch(self, root_directory=None):
        """
        Batch build knowledge graphs for all subdirectories under root directory.

        Args:
            root_directory: Root directory path. If None, uses TARGET_DIR from config

        Raises:
            ValueError: If root directory does not exist
        """
        if root_directory is None:
            root_directory = TARGET_DIR

        if not os.path.exists(root_directory):
            raise ValueError(f"Root directory does not exist: {root_directory}")

        subdirs = [os.path.join(root_directory, d)
                   for d in os.listdir(root_directory)
                   if os.path.isdir(os.path.join(root_directory, d))]

        for directory in tqdm(subdirs, desc="Building knowledge graphs"):
            await self.build_knowledge_graph_for_directory(directory)


async def main():
    """
    Main function to demonstrate knowledge graph building process.

    Creates a KnowledgeGraphBuilder instance and builds knowledge graphs
    for all directories under the configured TARGET_DIR.
    """
    builder = KnowledgeGraphBuilder()

    await builder.build_knowledge_graphs_batch()


if __name__ == '__main__':
    asyncio.run(main())