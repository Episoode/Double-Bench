import os
import base64
import asyncio
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm
import time
import concurrent.futures
import traceback
from bench_engine.config import SOURCE_DIR, TARGET_DIR, MAX_CONCURRENT_REQUESTS,QWEN_BASE_URL,QWEN_API_KEY,GT_MODEL
from bench_engine.prompt import PAGE_GT_PROMPT
from bench_engine.utils import async_load_json, async_save_json, load_openai_client, async_read_file


class GroundTruthProcessor:
    """
    Document processor class for document evidence page annotation.

    This class processes question-answer pairs and identifies which pages in documents
    contain evidence supporting the answers using LLM-based analysis.
    """

    def __init__(self,
                 input_file: str,
                 output_file: str,
                 model_name: str = GT_MODEL,
                 max_workers: int = 1,
                 semaphore_limit: int = MAX_CONCURRENT_REQUESTS):
        """
        Initialize the document processor.

        Args:
            input_file: Input JSON file path
            output_file: Output JSON file path
            model_name: Model name to use
            max_workers: Maximum number of worker threads in thread pool
            semaphore_limit: Async semaphore limit
        """
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.max_workers = max_workers
        self.semaphore_limit = semaphore_limit
        self.client = load_openai_client(async_mode=True,api_key=QWEN_API_KEY,base_url=QWEN_BASE_URL)

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self.all_results = []

    def load_image(self, image_path: str) -> Optional[str]:
        """
        Load image and convert to base64 encoding.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string or None if loading fails
        """
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"Image loading error {image_path}: {str(e)}")
            return None

    async def get_page_content(self, idx: int, doc_name: str) -> Dict[str, str]:
        """
        Get page OCR text content using async file reading from utils.

        Args:
            idx: Page index number
            doc_name: Document name

        Returns:
            Dictionary containing text content of the page
        """
        ocr_doc_folder_path = os.path.join(TARGET_DIR, doc_name)
        text_dir = os.path.join(ocr_doc_folder_path, "text")
        text_content = ""

        if os.path.exists(text_dir):
            text_file = os.path.join(text_dir, f"{idx}.txt")
            if os.path.exists(text_file):
                text_content = await async_read_file(text_file)
        return {"text": text_content}

    def get_base64_image(self, doc_name: str, page_num: int) -> Optional[str]:
        """
        Get base64 encoding of an image.

        Args:
            doc_name: Document name
            page_num: Page number

        Returns:
            Base64 encoded image string or None if not found
        """
        page_path = os.path.join(SOURCE_DIR, doc_name, f"{page_num}.jpg")
        if not os.path.exists(page_path):
            return None
        return self.load_image(page_path)

    async def process_single_image(self,
                                   doc_name: str,
                                   page_num: int,
                                   ocr_text: str,
                                   question: str,
                                   answer: str) -> Optional[str]:
        """
        Process single image and return model judgment result.

        Args:
            doc_name: Document name
            page_num: Page number
            ocr_text: OCR extracted text from the page
            question: Question text
            answer: Answer text

        Returns:
            Model response string or None if processing fails
        """
        base64_image = self.get_base64_image(doc_name, page_num)
        if not base64_image:
            print(f"Unable to get image {doc_name}/{page_num}.jpg")
            return None

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        if ocr_text:
            content.append({"type": "text", "text": f"Page content extracted by OCR:\n{ocr_text}"})
        content.append({"type": "text", "text": f"Query: {question}\nAnswer: {answer}"})

        messages = [
            {"role": "system", "content": PAGE_GT_PROMPT},
            {"role": "user", "content": content}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=10,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error occurred while processing image: {str(e)}")
            return None

    async def process_document_pages(self,
                                     doc_name: str,
                                     question: str,
                                     answer: str,
                                     semaphore: asyncio.Semaphore) -> Dict[int, str]:
        """
        Process all pages of a document.

        Args:
            doc_name: Document name
            question: Question text
            answer: Answer text
            semaphore: Semaphore for concurrency control

        Returns:
            Dictionary mapping page numbers to model responses
        """
        doc_images_path = os.path.join(SOURCE_DIR, doc_name)
        if not os.path.exists(doc_images_path):
            print(f"Document image directory does not exist: {doc_images_path}")
            return {}

        img_files = [f for f in os.listdir(doc_images_path) if f.endswith('.jpg')]
        if not img_files:
            return {}

        page_numbers = sorted([int(f.split('.')[0]) for f in img_files])
        results = {}

        async def process_page(page_num: int):
            async with semaphore:
                ocr_content = await self.get_page_content(page_num, doc_name)
                ocr_text = ocr_content.get("text", "")
                result = await self.process_single_image(
                    doc_name, page_num, ocr_text, question, answer
                )
                if result:
                    results[page_num] = result

        tasks = [process_page(p_num) for p_num in page_numbers]
        await asyncio.gather(*tasks)
        return results

    async def process_single_task(self,
                                  unit_metadata: dict,
                                  index: int,
                                  semaphore: asyncio.Semaphore) -> Optional[Dict]:
        """
        Process single task.

        Args:
            unit_metadata: Metadata for the unit to process
            index: Index of the unit
            semaphore: Semaphore for concurrency control

        Returns:
            Processed result dictionary or None if processing fails
        """
        original_uid = unit_metadata.get("uid", str(index))

        unit_question = unit_metadata["question"]
        unit_answer = unit_metadata["answer"]
        doc_name = unit_metadata["file_name"]
        original_evidence_page = unit_metadata.get("evidence_page", "")

        doc_images_path = os.path.join(SOURCE_DIR, doc_name)
        if not os.path.exists(doc_images_path):
            print(f"Document path does not exist: {doc_images_path} (UID: {original_uid})")
            return None

        page_results = await self.process_document_pages(doc_name, unit_question, unit_answer, semaphore)

        yes_pages = []
        for page_num, result_text in page_results.items():
            first_word = result_text.split()[0] if result_text and result_text.strip() else ""
            if first_word == "Yes":
                yes_pages.append(page_num)

        reference_pages = list(set(yes_pages))
        if original_evidence_page:
            try:
                if isinstance(original_evidence_page, str):
                    try:
                        original_page_num = int(original_evidence_page)
                        reference_pages.append(original_page_num)
                    except ValueError:
                        try:
                            page_nums = [int(p.strip()) for p in original_evidence_page.split(',') if p.strip()]
                            reference_pages.extend(page_nums)
                        except ValueError:
                            print(
                                f"Warning: UID {original_uid} evidence_page '{original_evidence_page}' cannot be parsed")
                elif isinstance(original_evidence_page, (int, float)):
                    reference_pages.append(int(original_evidence_page))
                elif isinstance(original_evidence_page, list):
                    for page in original_evidence_page:
                        try:
                            reference_pages.append(int(page))
                        except (ValueError, TypeError):
                            print(
                                f"Warning: UID {original_uid} evidence_page value '{page}' cannot be converted to integer")
            except Exception as e:
                print(f"Warning: UID {original_uid} error processing evidence_page: {str(e)}")

        reference_pages = sorted(list(set(reference_pages)))

        return {
            "uid": original_uid,
            "question": unit_question,
            "answer": unit_answer,
            "modality": unit_metadata.get("modality", ""),
            "file_name": doc_name,
            "reference_page": reference_pages,
        }

    def preprocess_metadata(self, metadata_list: List[Dict]) -> Dict[str, List]:
        """
        Group metadata by document name to improve processing efficiency.

        Args:
            metadata_list: List of metadata dictionaries

        Returns:
            Dictionary grouping metadata by document name
        """
        doc_groups = {}
        for i, meta in enumerate(metadata_list):
            doc_name = meta["file_name"]

            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append((i, meta))
        return doc_groups

    async def process_doc_group(self,
                                doc_name: str,
                                items: List,
                                semaphore: asyncio.Semaphore,
                                pbar: tqdm) -> List[Dict]:
        """
        Process all questions for the same document.

        Args:
            doc_name: Document name
            items: List of items to process for this document
            semaphore: Semaphore for concurrency control
            pbar: Progress bar object

        Returns:
            List of processed results
        """
        tasks = []
        for original_idx, meta in items:
            task = asyncio.create_task(self.process_single_task(meta, original_idx, semaphore))
            tasks.append(task)

        group_results = []
        if tasks:
            group_results = await asyncio.gather(*tasks)

        valid_results = [r for r in group_results if r is not None]

        if valid_results:
            self.all_results.extend(valid_results)

        pbar.update(len(items))

        return valid_results

    async def process_async(self):
        """
        Main async processing function.

        Loads input data, processes all documents and their questions,
        and manages the overall workflow with progress tracking.
        """
        gt_json = await async_load_json(self.input_file)
        if gt_json is None:
            print(f"Failed to load input file JSON data")
            return

        doc_groups = self.preprocess_metadata(gt_json)
        print(f"Grouped {len(gt_json)} questions into {len(doc_groups)} document groups")

        pbar = tqdm(total=len(doc_groups), desc="Processing document groups")
        semaphore = asyncio.Semaphore(self.semaphore_limit)

        self.all_results = []

        for doc_name, items_in_group in doc_groups.items():
            await self.process_doc_group(
                doc_name, items_in_group, semaphore, pbar
            )

        pbar.close()
        print(f"\nProcessing completed, processed {len(self.all_results)} results in total")

    def process(self):
        """
        Main processing function entry point.

        Executes the complete processing workflow including timing and error handling,
        ensuring results are saved even if errors occur during processing.
        """
        time1 = time.time()

        try:
            asyncio.run(self.process_async())
        except Exception as e:
            print(f"Error occurred during execution: {str(e)}")
            traceback.print_exc()
        finally:
            time2 = time.time()
            print(f"Total time elapsed: {time2 - time1:.2f} seconds")

            if self.all_results:
                print(f"Saving {len(self.all_results)} results to {self.output_file}...")
                try:
                    asyncio.run(async_save_json(self.all_results, self.output_file))
                    print("Results saved successfully.")
                except Exception as e:
                    print(f"Failed to save results: {str(e)}")
            else:
                print("No results to save.")

