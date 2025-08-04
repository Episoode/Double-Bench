import json
import os
import random
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from typing import List, Optional
from bench_engine.utils import load_openai_client, async_read_file, async_load_image, async_save_json
from bench_engine.prompt import SINGLEHOP_QA_TEXT_PROMPT, SINGLEHOP_QA_IMG_PROMPT
from bench_engine.config import MAX_CONCURRENT_REQUESTS, QA_GENERATE_MODEL, TARGET_DIR, SINGLEHOP_QA_JSON_PATH


class SinglehopQAGenerator:
    """
    Single-hop QA dataset generator class.
    Used for processing multimodal documents and generating high-quality question-answer pair datasets.
    """

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """
        Initialize the single-hop QA dataset generator.

        Args:
            client (Optional[AsyncOpenAI]): OpenAI async client, automatically created if None
        """
        self.client = client or load_openai_client(async_mode=True)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.text_prompt = SINGLEHOP_QA_TEXT_PROMPT
        self.img_prompt = SINGLEHOP_QA_IMG_PROMPT
        self.model = QA_GENERATE_MODEL
        self.target_dir = TARGET_DIR
        self.output_json_path = SINGLEHOP_QA_JSON_PATH

    def _parse_json_response(self, response_content: str) -> Optional[dict]:
        """
        Parse JSON response content.

        Args:
            response_content (str): Response content string

        Returns:
            Optional[dict]: Parsed dictionary or None if parsing fails
        """
        try:
            # Try to parse JSON directly
            return json.loads(response_content)
        except json.JSONDecodeError:
            try:
                # Try to clean content and reparse
                cleaned_content = response_content.strip()
                if cleaned_content.startswith('```json'):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith('```'):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Original content: {response_content}")
                return None

    async def _generate_text_qa(self, doc_dir: str, text_dir: str, selected_file: str) -> Optional[dict]:
        """
        Generate QA pairs for text modality.

        Args:
            doc_dir (str): Document directory path
            text_dir (str): Text directory path
            selected_file (str): Selected file name

        Returns:
            Optional[dict]: Generated QA pair dictionary or None
        """
        evidence = os.path.splitext(selected_file)[0]
        txt_path = os.path.join(text_dir, selected_file)
        content = await async_read_file(txt_path)

        if len(content) < 100:
            return None

        messages = [
            {
                "role": "system",
                "content": '''You are a professional cross-document retrieval dataset assistant. 
                Read the information on the given page and generate a high-quality QA pair in JSON format.
                ''' + self.text_prompt
            },
            {
                "role": "user",
                "content": f"Content:\n{content}"
            }
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        json_dict = self._parse_json_response(response.choices[0].message.content)
        if json_dict is None:
            return None

        json_dict.update({
            "modality": "text",
            "file_name": os.path.basename(doc_dir),
            "evidence_page": str(evidence)
        })
        return json_dict

    async def _generate_image_qa(self, doc_dir: str, text_dir: str, modality_dir: str,
                                 modality: str, selected_file: str) -> Optional[dict]:
        """
        Generate QA pairs for image modality.

        Args:
            doc_dir (str): Document directory path
            text_dir (str): Text directory path
            modality_dir (str): Modality directory path
            modality (str): Modality type (figure or table)
            selected_file (str): Selected file name

        Returns:
            Optional[dict]: Generated QA pair dictionary or None
        """
        img_name = selected_file
        evidence = int(img_name.split('_')[0])
        img_path = os.path.join(modality_dir, img_name)

        # Use async_load_image function from utils
        img_data_url = await async_load_image(img_path, 'png')
        if img_data_url is None:
            return None

        messages = [
            {
                "role": "system",
                "content": '''
                You are an expert visual data analyst specializing in cross-modal retrieval dataset construction. 
                Analyze the given image and its page context to generate one high-quality QA pair in JSON format.
                ''' + self.img_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img_data_url, "detail": "high"}
                    },
                ]
            }
        ]

        # Add context information (if exists)
        context_path = os.path.join(text_dir, f"{evidence}.txt")
        if await asyncio.to_thread(os.path.exists, context_path):
            context = await async_read_file(context_path)
            messages[1]["content"].append({
                "type": "text",
                "text": f'context:\n{context}'
            })

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        json_dict = self._parse_json_response(response.choices[0].message.content)
        if json_dict is None:
            return None

        json_dict.update({
            "modality": modality,
            "file_name": os.path.basename(doc_dir),
            "evidence_page": str(evidence)
        })
        return json_dict

    async def _generate_single_modality_qa_json(self, doc_dir: str, text_dir: str,
                                                modality_dir: str, modality: str,
                                                selected_file: str) -> Optional[dict]:
        """
        Generate QA pairs for a single modality.

        Args:
            doc_dir (str): Document directory path
            text_dir (str): Text directory path
            modality_dir (str): Modality directory path
            modality (str): Modality type
            selected_file (str): Selected file name

        Returns:
            Optional[dict]: Generated QA pair dictionary or None
        """
        async with self.semaphore:
            try:
                if modality == 'text':
                    return await self._generate_text_qa(doc_dir, text_dir, selected_file)
                else:
                    return await self._generate_image_qa(doc_dir, text_dir, modality_dir,
                                                         modality, selected_file)
            except Exception as e:
                print(f"Error in {modality} modality QA generation: {str(e)}")
                return None

    async def _extract_page_ids_from_files(self, directory_path: str) -> set:
        """
        Extract all page IDs (numbers before underscore) from file names in directory.

        Args:
            directory_path (str): Directory path

        Returns:
            set: Set of page IDs
        """
        if not await asyncio.to_thread(os.path.exists, directory_path):
            return set()

        files = await asyncio.to_thread(os.listdir, directory_path)
        page_ids = set()

        for file in files:
            try:
                page_id = file.split('_')[0]
                if page_id.isdigit():
                    page_ids.add(page_id)
            except (IndexError, ValueError):
                continue

        return page_ids

    async def _generate_doc_question(self, doc_dir: str) -> List[dict]:
        """
        Generate QA pairs for a single document.

        Args:
            doc_dir (str): Document directory path

        Returns:
            List[dict]: List of QA pairs
        """
        try:
            data = []
            text_dir = os.path.join(doc_dir, 'text')
            modalities = ['text']

            # Check if figure and table modalities exist
            for mod in ['table', 'figure']:
                mod_path = os.path.join(doc_dir, mod)
                if await asyncio.to_thread(os.path.exists, mod_path):
                    if len(await asyncio.to_thread(os.listdir, mod_path)) > 0:
                        modalities.append(mod)

            # Collect all page IDs from figures and tables
            excluded_page_ids = set()
            for mod in ['table', 'figure']:
                mod_path = os.path.join(doc_dir, mod)
                mod_page_ids = await self._extract_page_ids_from_files(mod_path)
                excluded_page_ids.update(mod_page_ids)

            # Process text modality
            all_text_files = await asyncio.to_thread(os.listdir, text_dir)

            # Filter out pages containing figures and tables
            filtered_text_files = []
            for file in all_text_files:
                page_id = os.path.splitext(file)[0]
                if page_id not in excluded_page_ids:
                    filtered_text_files.append(file)

            # Use all text files if not enough filtered files
            if not filtered_text_files:
                print(f"Warning: No pages without figures/tables in {doc_dir}, using all pages")
                filtered_text_files = all_text_files

            # Randomly select specified number of text files
            text_num = random.randint(2, min(5, len(filtered_text_files)))
            selected_text_files = random.sample(filtered_text_files, min(text_num, len(filtered_text_files)))

            # Create text QA generation tasks
            text_tasks = [
                self._generate_single_modality_qa_json(
                    doc_dir, text_dir, None, 'text', file
                ) for file in selected_text_files
            ]
            text_results = await asyncio.gather(*text_tasks)
            data.extend([res for res in text_results if res])

            # Process other modalities (figures and tables)
            for modality in modalities[1:]:
                mod_path = os.path.join(doc_dir, modality)
                mod_files = await asyncio.to_thread(os.listdir, mod_path)

                num = min(random.randint(1, 3), len(mod_files))

                # Randomly select modality files
                selected_mod_files = random.sample(mod_files, min(num, len(mod_files))) if mod_files else []
                mod_tasks = [
                    self._generate_single_modality_qa_json(
                        doc_dir, text_dir, mod_path, modality, file
                    ) for file in selected_mod_files
                ]
                mod_results = await asyncio.gather(*mod_tasks)
                data.extend([res for res in mod_results if res])

            return data

        except Exception as e:
            print(f"Error processing document {doc_dir}: {str(e)}")
            return []

    async def generate_base_bench(self) -> str:
        """
        Generate benchmark dataset.

        Returns:
            str: Output file path
        """
        docs = await asyncio.to_thread(os.listdir, self.target_dir)
        total_docs = len(docs)

        print(f"Starting to process {total_docs} documents in total...")
        print(f"Target directory: {self.target_dir}")
        print(f"Output file: {self.output_json_path}")

        # Create tasks for each document
        tasks = [
            self._generate_doc_question(os.path.join(self.target_dir, doc))
            for doc in docs
        ]
        all_results = await tqdm_asyncio.gather(
            *tasks,
            desc="Document processing progress",
            total=total_docs,
        )

        final_data = []
        successful = 0
        failed = 0

        for result in all_results:
            if isinstance(result, Exception):
                print(f"Document processing error: {result}")
                failed += 1
            elif result:
                final_data.extend(result)
                successful += 1

        print(f"\nProcessing completed!")
        print(f"Successfully processed documents: {successful}")
        print(f"Failed documents: {failed}")
        print(f"Total QA pairs generated: {len(final_data)}")

        # Use async_save_json function from utils to save results
        await async_save_json(final_data, self.output_json_path)
        print(f"Results saved to {self.output_json_path}")
        return self.output_json_path


# Usage example
async def main():
    """Main function example."""
    generator = SinglehopQAGenerator()
    await generator.generate_base_bench()


if __name__ == "__main__":
    asyncio.run(main())