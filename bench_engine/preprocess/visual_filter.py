import os
import asyncio
from PIL import Image
import aiofiles
import io
from bench_engine.prompt import VISUAL_FILTER_SYSTEM_PROMPT
from bench_engine.config import MIN_IMAGE_SIZE, MAX_CONCURRENT_REQUESTS
from bench_engine.utils import async_read_file, load_openai_client, async_encode_images_batch


class VisualFliter:
    """
    A visual filter class that processes images using VLM to determine
    if images should be kept or discarded based on visual content analysis.
    """

    def __init__(self):
        """
        Initialize the VisualFliter instance.
        Sets up the OpenAI client and system prompt for image filtering.
        """
        self.client = load_openai_client(async_mode=True)
        self.sys_prompt = VISUAL_FILTER_SYSTEM_PROMPT

    async def check_image_size(self, img_path):
        """
        Asynchronously check if an image meets the minimum size requirements.

        Args:
            img_path (str): Path to the image file to be checked

        Returns:
            tuple: A tuple containing (is_valid: bool, img_data: bytes)
                   is_valid - True if image meets size requirements, False otherwise
                   img_data - Raw image data as bytes
        """
        async with aiofiles.open(img_path, 'rb') as f:
            img_data = await f.read()
            img = Image.open(io.BytesIO(img_data))
            width, height = img.size
            return width >= MIN_IMAGE_SIZE and height >= MIN_IMAGE_SIZE, img_data

    async def encode2base64(self, img_paths):
        """
        Process multiple images for encoding, utilizing batch encoding from utils.

        Args:
            img_paths (list): List of image file paths to be processed

        Returns:
            tuple: A tuple containing (base64_results: list, valid_images: list, invalid_images: list)
                   base64_results - List of base64 encoded image strings
                   valid_images - List of valid image paths that passed size check
                   invalid_images - List of invalid image paths that failed size check
        """
        tasks = [self.check_image_size(img_path) for img_path in img_paths]
        results = await asyncio.gather(*tasks)

        valid_images = []
        invalid_images = []

        for img_path, (is_valid, img_data) in zip(img_paths, results):
            if is_valid:
                valid_images.append(img_path)
            else:
                invalid_images.append(img_path)

        if valid_images:
            base64_results = await async_encode_images_batch(valid_images)
        else:
            base64_results = []

        return base64_results, valid_images, invalid_images

    def group_files_by_first_digit(self, directory):
        """
        Group files in a directory by the first digit in their filename (before underscore).

        Args:
            directory (str): Path to the directory containing files to be grouped

        Returns:
            list: A list of file groups, where each group contains files with the same first digit
                  Each group is a list of full file paths
        """
        grouped_files = {}
        for filename in os.listdir(directory):
            name_without_extension = os.path.splitext(filename)[0]
            if "_" in name_without_extension:
                first_digit = int(name_without_extension.split("_")[0])
                if first_digit not in grouped_files:
                    grouped_files[first_digit] = []
                grouped_files[first_digit].append(os.path.join(directory, filename))
        return list(grouped_files.values())

    async def read_text_file(self, text_path):
        """
        Asynchronously read a text file.

        Args:
            text_path (str): Path to the text file to be read

        Returns:
            str: Content of the text file, or empty string if file doesn't exist
        """
        if not os.path.exists(text_path):
            return ''
        return await async_read_file(text_path)

    async def process_single_page(self, imgs, text):
        """
        Process a single page with images and text using GPT-4V model.

        Args:
            imgs (list): List of base64 encoded image strings
            text (str): Text content associated with the images

        Returns:
            str: The model's response indicating whether to keep ('Yes') or discard ('No') the content
        """
        response = await self.client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {
                    'role': 'system',
                    'content': self.sys_prompt
                },
                {
                    'role': 'user',
                    "content": [
                        {"type": "text",
                         "text": 'context:\n' + text
                         },
                        *[{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}", "detail": "high"}
                        } for img_base64 in imgs],
                    ]
                }
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.split()[0]

    async def move_files(self, imgs, discard_dir):
        """
        Move image files to the discard directory.

        Args:
            imgs (list): List of image file paths to be moved
            discard_dir (str): Path to the discard directory where files will be moved
        """
        for img in imgs:
            img_name = os.path.basename(img)
            os.rename(img, os.path.join(discard_dir, img_name))

    async def process_page_group(self, folder_dir, text_dir, discard_dir, imgs):
        """
        Process a group of images belonging to the same page.

        Args:
            folder_dir (str): Path to the folder containing image files
            text_dir (str): Path to the directory containing text files
            discard_dir (str): Path to the directory for discarded files
            imgs (list): List of image file paths belonging to the same page
        """
        page_idx = os.path.basename(imgs[0]).split('_')[0]

        text_path = os.path.join(text_dir, f'{page_idx}.txt')
        text_task = self.read_text_file(text_path)
        base64_task = self.encode2base64(imgs)

        text, (base64_imgs, valid_imgs, invalid_imgs) = await asyncio.gather(text_task, base64_task)

        if invalid_imgs:
            await self.move_files(invalid_imgs, discard_dir)

        if not valid_imgs:
            return

        response = await self.process_single_page(base64_imgs, text)

        if response == 'No':
            await self.move_files(valid_imgs, discard_dir)
        elif response not in ['Yes', 'No']:
            print('Error')
            print(response)
            print('Error')

    async def process_modality(self, doc_dir, modality, semaphore):
        """
        Process all images for a single modality (figure or table) within a document directory.

        Args:
            doc_dir (str): Path to the document directory
            modality (str): Type of modality to process ('figure' or 'table')
            semaphore (asyncio.Semaphore): Semaphore to control concurrency
        """
        discard_dir = os.path.join(doc_dir, f'discard_{modality}')
        os.makedirs(discard_dir, exist_ok=True)
        folder_dir = os.path.join(doc_dir, modality)
        text_dir = os.path.join(doc_dir, 'text')

        page_imgs = self.group_files_by_first_digit(folder_dir)

        async def bounded_process(imgs):
            async with semaphore:
                await self.process_page_group(folder_dir, text_dir, discard_dir, imgs)

        tasks = [bounded_process(imgs) for imgs in page_imgs]
        await asyncio.gather(*tasks)

    async def process_dir(self, doc_dir, semaphore):
        """
        Concurrently process both figure and table modalities for a document directory.

        Args:
            doc_dir (str): Path to the document directory to be processed
            semaphore (asyncio.Semaphore): Semaphore to control concurrency across modalities
        """
        tasks = [
            self.process_modality(doc_dir, modality, semaphore)
            for modality in ['figure', 'table']
        ]
        await asyncio.gather(*tasks)

    async def filter_images(self, dir_path, docs):
        """
        Main image filtering method that processes multiple documents concurrently.

        Args:
            dir_path (str): Base directory path containing all document folders
            docs (list): List of document folder names to be processed
        """
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def process_doc(doc):
            doc_dir = os.path.join(dir_path, doc)
            await self.process_dir(doc_dir, semaphore)

        await asyncio.gather(*[process_doc(doc) for doc in docs])