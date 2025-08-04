import os
from pathlib import Path
from docling_core.types.doc import PictureItem, TableItem
import asyncio
from tqdm import tqdm
from bench_engine.utils import load_converter, async_write_file


class DocumentParser:
    def __init__(self, scale=2.0):
        """
        Initialize document parser with specified scale
        Args:
            scale (float): Scaling factor for document processing, default 2.0
        """
        self.converter = load_converter(scale)

    async def process_single_page(self, input_doc_dir, page, text_dir, table_dir, figure_dir):
        """
        Process a single document page and extract text, tables, and figures
        Args:
            input_doc_dir (str): Input document directory path
            page (str): Page filename to process
            text_dir (str): Output directory for text files
            table_dir (str): Output directory for table images
            figure_dir (str): Output directory for figure images
        """
        conv_res = self.converter.convert(os.path.join(input_doc_dir, page))
        full_doc_md = conv_res.document.export_to_markdown()

        table_counter = 0
        picture_counter = 0
        page_name = page.split('.jpg')[0]

        await async_write_file(os.path.join(text_dir, f"{page_name}.txt"), full_doc_md)

        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = Path(os.path.join(table_dir, f"{page_name}_{table_counter - 1}.png"))
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")

            if isinstance(element, PictureItem):
                picture_counter += 1
                element_image_filename = Path(os.path.join(figure_dir, f"{page_name}_{picture_counter - 1}.png"))
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")

    async def process_doc(self, input_doc_dir: str, output_doc_dir: str):
        """
        Process entire document directory and organize outputs
        Args:
            input_doc_dir (str): Input document directory path
            output_doc_dir (str): Output directory for processed files
        """
        os.makedirs(output_doc_dir, exist_ok=True)
        text_dir = os.path.join(output_doc_dir, 'text')
        table_dir = os.path.join(output_doc_dir, 'table')
        figure_dir = os.path.join(output_doc_dir, 'figure')

        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(table_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

        pages = os.listdir(input_doc_dir)
        tasks = []
        for page in tqdm(pages):
            task = self.process_single_page(input_doc_dir, page, text_dir, table_dir, figure_dir)
            tasks.append(task)

        await asyncio.gather(*tasks)