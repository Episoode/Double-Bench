import asyncio
import json
import os
import re
import time
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Dict, Any, Optional

from bench_engine.prompt import SINGLEHOP_TEXT_JUDGE, SINGLEHOP_IMG_JUDGE
from bench_engine.config import MAX_CONCURRENT_REQUESTS, QA_JUDGE_MODEL, SINGLEHOP_QA_JSON_PATH, \
    SINGLEHOP_JUDGED_JSON_PATH, \
    TARGET_DIR
from bench_engine.utils import load_openai_client, async_read_file, async_encode_images_batch, async_load_json, \
    async_save_json


class SinglehopQAJudger:
    """
    Single-hop QA pair judger class.
    Used for filtering and evaluating the quality of QA pairs to ensure they meet retrieval system requirements.
    """

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """
        Initialize the single-hop QA judger.

        Args:
            client (Optional[AsyncOpenAI]): OpenAI async client, creates one using environment API_KEY if None
        """
        self.client = client or load_openai_client(async_mode=True)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.text_prompt = SINGLEHOP_TEXT_JUDGE
        self.img_prompt = SINGLEHOP_IMG_JUDGE
        self.model = QA_JUDGE_MODEL
        self.input_json_path = SINGLEHOP_QA_JSON_PATH
        self.output_json_path = SINGLEHOP_JUDGED_JSON_PATH
        self.target_dir = TARGET_DIR

    def _replace_markdown_table(self, content: str) -> str:
        """
        Replace Markdown tables with simple markers.

        Args:
            content (str): Input text content

        Returns:
            str: Processed text content with tables replaced
        """
        pattern = r'\|[^\n]*\|(\n\|[^\n]*\|)*'
        result = re.sub(pattern, '<table>', content)
        return result

    def _find_matching_images(self, folder_path: str, idx: str) -> List[str]:
        """
        Find image files matching a specific index.

        Args:
            folder_path (str): Path to the image folder
            idx (str): Page index to match

        Returns:
            List[str]: List of matching image file names
        """
        pattern = re.compile(rf'^{re.escape(str(idx))}_.+\.png$', re.IGNORECASE)
        matched_files = []
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and pattern.match(entry.name):
                    matched_files.append(entry.name)
        return matched_files

    def _parse_judge_result(self, response_content: str) -> str:
        """
        Parse judge result JSON and extract the keep field.

        Args:
            response_content (str): Content returned by API

        Returns:
            str: "Yes" or "No"
        """
        try:
            # Try to parse JSON
            result = json.loads(response_content.strip())

            # Extract keep field with case-insensitive handling
            keep_value = result.get('keep', 'No')
            if isinstance(keep_value, str):
                return "Yes" if keep_value.lower() == "yes" else "No"
            else:
                return "No"
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, try direct Yes/No matching
            content = response_content.strip().lower()
            if 'yes' in content:
                return "Yes"
            else:
                return "No"

    async def _judge_qa_pair_async(self, item: Dict[str, Any]) -> str:
        """
        Asynchronously judge the quality of a single QA pair.

        Args:
            item (Dict[str, Any]): QA pair data item (new format)

        Returns:
            str: Judge result ("Yes" or "No")
        """
        async with self.semaphore:
            try:
                # Adapt to new JSON format
                doc_dir = os.path.join(self.target_dir, item['file_name'])
                idx = item['evidence_page']
                text_path = os.path.join(doc_dir, 'text', idx + '.txt')

                # Use async_read_file function from utils to read text content
                text_content = await async_read_file(text_path)

                # Use new field names
                question = item['question']
                answer = item['answer']
                modality = item['modality']

                if modality != 'text':
                    # Handle cases with images
                    img_dir = os.path.join(doc_dir, 'figure')
                    page_images = self._find_matching_images(img_dir, idx)
                    img_paths = [os.path.join(img_dir, img) for img in page_images]

                    # Use async_encode_images_batch function from utils to encode images asynchronously
                    base64_imgs = await async_encode_images_batch(img_paths)
                    text_content = self._replace_markdown_table(text_content)

                    # Build messages with images
                    messages = [
                        {"role": "system", "content": self.img_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": f"Query:\n{question}\n Answer:\n{answer}"},
                                {"type": "text", "text": 'context:\n' + text_content}
                            ]
                        }
                    ]

                    # Add images to messages
                    for base64_img in base64_imgs:
                        messages[1]["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img}",
                                "detail": "high"
                            }
                        })
                else:
                    # Handle text-only cases
                    messages = [
                        {"role": "system", "content": self.text_prompt},
                        {
                            "role": "user",
                            "content": f"\n\nQuery:\n{question}\n Answer:\n{answer}\n\ncontext:\n" + text_content
                        }
                    ]

                # Call OpenAI API
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},  # Add JSON format requirement
                    max_tokens=150,  # Increase token count to accommodate reason field
                    temperature=0
                )

                # Parse judge result
                return self._parse_judge_result(response.choices[0].message.content)

            except Exception as e:
                print(f"Error processing item: {str(e)}, returning default value 'No'")
                return "No"

    async def judge_qa_pairs_async(self) -> str:
        """
        Asynchronously judge the quality of multiple QA pairs, only save items judged as Yes, and assign uid to each item.

        Returns:
            str: Output file path
        """
        data = await async_load_json(self.input_json_path)
        if data is None:
            print(f"Unable to read input file: {self.input_json_path}")
            return self.output_json_path

        # Create all tasks
        tasks = []
        for i, item in enumerate(data):
            task = self._judge_qa_pair_async(item)
            tasks.append(task)

        # Execute all tasks asynchronously with progress tracking
        results = []
        for task in async_tqdm.as_completed(tasks, total=len(tasks), desc="Processing QA pairs"):
            result = await task
            results.append(result)

        # Keep only items judged as Yes and assign uid to each item
        kept_data = []
        uid_counter = 0
        for i, result in enumerate(results):
            if result == "Yes":
                # Create new dictionary and add uid field
                item_with_uid = data[i].copy()
                item_with_uid['uid'] = str(uid_counter)
                kept_data.append(item_with_uid)
                uid_counter += 1

        # Use async_save_json function from utils to save results
        await async_save_json(kept_data, self.output_json_path)

        return self.output_json_path

    async def judge_dataset(self) -> Dict[str, Any]:
        """
        Judge dataset and return statistical information.

        Returns:
            Dict[str, Any]: Dictionary containing statistical information
        """
        start_time = time.time()

        print(f"Starting QA pair judging task...")
        print(f"Target directory: {self.target_dir}")
        print(f"Input file: {self.input_json_path}")
        print(f"Output file: {self.output_json_path}")

        # Use async_load_json function from utils to read original data and get total count
        original_data = await async_load_json(self.input_json_path)
        if original_data is None:
            print(f"Unable to read input file: {self.input_json_path}")
            return {}

        total_count = len(original_data)

        output_path = await self.judge_qa_pairs_async()

        # Use async_load_json function from utils to count results
        kept_data = await async_load_json(output_path)
        if kept_data is None:
            print(f"Unable to read output file: {output_path}")
            return {}

        keep_count = len(kept_data)
        reject_count = total_count - keep_count

        end_time = time.time()

        stats = {
            'total_qa_pairs': total_count,
            'kept_pairs': keep_count,
            'rejected_pairs': reject_count,
            'keep_rate': keep_count / total_count if total_count > 0 else 0,
            'processing_time': end_time - start_time,
            'target_dir': self.target_dir,
            'input_file': self.input_json_path,
            'output_file': self.output_json_path
        }

        print(f"Processing completed! Results saved to: {output_path}")
        print(f"Total QA pairs: {total_count}")
        print(f"Kept QA pairs: {keep_count}")
        print(f"Rejected QA pairs: {reject_count}")
        print(f"Keep rate: {stats['keep_rate']:.2%}")
        print(f"Total time: {stats['processing_time']:.2f} seconds")

        return stats


# Usage example
async def main():
    """Main function example."""
    judger_instance = SinglehopQAJudger()

    # Execute judging task
    stats = await judger_instance.judge_dataset()

    return stats


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run main function
    asyncio.run(main())