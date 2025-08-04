import asyncio
import time
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

from bench_engine.config import MAX_CONCURRENT_REQUESTS, TARGET_DIR, CAPTION_MODEL
from bench_engine.prompt import TABLE_CAPTION_PROMPT, FIGURE_CAPTION_PROMPT
from bench_engine.utils import async_write_file, async_load_image, load_openai_client


class VisualCaptionProcessor:
    """
    Visual caption processor class for processing document images and generating descriptions.

    This class handles image processing workflows including table and figure caption generation
    using OpenAI's vision API, with support for concurrent processing and progress tracking.
    """

    def __init__(self):
        """
        Initialize the visual caption processor.

        Sets up the base directory, concurrency limits, model configuration,
        and initializes the OpenAI client for image processing tasks.
        """
        self.base_dir = Path(TARGET_DIR)
        self.max_concurrent_tasks = MAX_CONCURRENT_REQUESTS
        self.model = CAPTION_MODEL
        self.progress_lock = asyncio.Lock()

        self.client = load_openai_client(async_mode=True)

    async def save_text(self, text: str, output_path: Path) -> bool:
        """
        Save text to specified path.

        Args:
            text: Text content to save
            output_path: Path where the text file should be saved

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await async_write_file(str(output_path), text)
            return True
        except Exception:
            print(f"Failed to save text to {output_path}")
            return False

    def get_output_path(self, image_path: Path, img_type: str) -> Path:
        """
        Get output file path for processed image.

        Args:
            image_path: Original image file path
            img_type: Image type ("table" or "figure")

        Returns:
            Path: Output path for the text file with modified directory structure
        """
        rel_path = image_path.relative_to(self.base_dir)

        parts = list(rel_path.parts)
        type_idx = next((i for i, part in enumerate(parts) if part == img_type), None)
        if type_idx is not None:
            parts[type_idx] = f"{img_type}_text"

        return self.base_dir.joinpath(*parts).with_suffix('.txt')

    async def generate_image_description(self, image_path: Path, img_type: str) -> bool:
        """
        Generate and save image description using OpenAI API.

        Args:
            image_path: Path to the image file
            img_type: Image type ("table" or "figure")

        Returns:
            bool: True if successful, False if failed
        """
        output_path = self.get_output_path(image_path, img_type)
        if output_path.exists():
            return True

        image_data_url = await async_load_image(str(image_path), 'png')
        if not image_data_url:
            return False

        prompt = TABLE_CAPTION_PROMPT if img_type == "table" else FIGURE_CAPTION_PROMPT

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": image_data_url,
                        }}
                    ]}
                ],
                max_tokens=1000,
                temperature=0
            )

            description = response.choices[0].message.content
            success = await self.save_text(description, output_path)
            return success
        except Exception as e:
            print(f"Failed to generate image description for {image_path}: {e}")
            return False

    async def scan_document(self, doc_path: Path) -> Dict[str, List[Path]]:
        """
        Scan single document directory to find all images that need processing.

        Args:
            doc_path: Document directory path

        Returns:
            Dictionary with image types as keys and lists of image paths as values
        """
        image_files = {"table": [], "figure": []}

        for img_type in ["table", "figure"]:
            type_dir = doc_path / img_type
            if not type_dir.exists() or not type_dir.is_dir():
                continue

            for img_file in type_dir.glob("**/*.png"):
                if img_file.is_file():
                    image_files[img_type].append(img_file)

        return image_files

    async def scan_all_documents(self) -> List[Tuple[Path, str]]:
        """
        Scan all document directories.

        Returns:
            List of tuples containing document path and document ID
        """
        documents = []

        if not self.base_dir.exists():
            print(f"Error: Base directory does not exist: {self.base_dir}")
            return documents

        for doc_dir in self.base_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name
            documents.append((doc_dir, doc_id))

        return documents

    async def process_single_document(self, doc_path: Path, doc_id: str, progress_bar=None) -> bool:
        """
        Process all images in a single document.

        Args:
            doc_path: Document path
            doc_id: Document ID
            progress_bar: Total progress bar object

        Returns:
            bool: Whether processing was successful
        """
        image_files = await self.scan_document(doc_path)
        total_images = sum(len(files) for files in image_files.values())

        if total_images == 0:
            if progress_bar:
                async with self.progress_lock:
                    progress_bar.update(1)
            return False

        tasks = []
        for img_type, files in image_files.items():
            for img_path in files:
                tasks.append(self.generate_image_description(img_path, img_type))

        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        async def limited_task(task_coroutine):
            async with semaphore:
                return await task_coroutine

        results = await asyncio.gather(*[limited_task(task) for task in tasks])

        success_count = sum(1 for result in results if result)
        fail_count = len(results) - success_count

        if progress_bar:
            async with self.progress_lock:
                progress_bar.update(1)

        return success_count == total_images

    async def process_all_documents(self):
        """
        Process all documents in the base directory.

        Scans for all document directories and processes their images concurrently,
        providing progress tracking and success/failure statistics.
        """
        documents = await self.scan_all_documents()
        total_docs = len(documents)
        if not documents:
            print("No documents found.")
            return
        print(f"Found {total_docs} documents to process")

        processed_count = 0
        failed_count = 0

        progress_bar = tqdm(total=total_docs, desc="Document processing progress", unit="docs")

        try:
            for doc_path, doc_id in documents:
                success = await self.process_single_document(doc_path, doc_id, progress_bar)

                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    print(f"Document processing failed: {doc_id}")
        finally:
            progress_bar.close()

        print(f"All documents processed! Successful: {processed_count}, Failed: {failed_count}")

    async def run(self):
        """
        Run the main processing workflow.

        Executes the complete document processing pipeline including initialization,
        processing all documents, and reporting final statistics with timing information.
        """
        print("Starting document directory processing...")
        start_time = time.time()
        await self.process_all_documents()
        elapsed = time.time() - start_time
        print(f"All processing completed! Total time: {elapsed:.2f} seconds")


async def main():
    """
    Main function to run the visual caption processing workflow.

    Creates a VisualCaptionProcessor instance and executes the complete
    image processing pipeline for all documents in the configured directory.
    """
    processor = VisualCaptionProcessor()

    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())