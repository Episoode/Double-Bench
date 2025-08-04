import asyncio
import base64
import io
import json
import warnings
from PIL import Image
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from openai import OpenAI, AsyncClient
from bench_engine.config import OPENAI_API_KEY, DEFAULT_SCALE, OPENAI_BASE_URL

warnings.filterwarnings("ignore")

async def async_read_file(file_path):
    try:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None,
            lambda: open(file_path, 'r', encoding='utf-8').read()
        )
        return content
    except Exception as e:
        print(f"Error occurred while reading file {file_path}: {str(e)}")
        return ""

async def async_write_file(file_path, content):
    """异步写入文件"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: open(file_path, 'w', encoding='utf-8').write(content)
        )
    except Exception as e:
        print(f"Error occurred while writing file {file_path}: {str(e)}")

async def async_load_image(img_path, format_type='png'):
    try:
        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(None, lambda: Image.open(img_path))
        img_str = await loop.run_in_executor(None, lambda: pil_images_to_base64(img, format_type))
        return f"data:image/{format_type};base64,{img_str}"
    except Exception as e:
        print(f"Error occurred while loading image {img_path}: {str(e)}")
        return None

def load_openai_client(async_mode=True,api_key = OPENAI_API_KEY,base_url=OPENAI_BASE_URL):
    if async_mode:
        return AsyncClient(api_key=api_key,base_url=base_url)
    return OpenAI(api_key=api_key,base_url=base_url)

async def async_load_json(file_path):
    """异步读取JSON文件"""
    try:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None,
            lambda: open(file_path, 'r', encoding='utf-8').read()
        )
        return json.loads(content)
    except Exception as e:
        print(f"Error occurred while loading JSON file {file_path}: {str(e)}")
        return None

async def async_save_json(data, file_path, indent=4, ensure_ascii=False):
    """异步保存JSON文件"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: json.dump(data, open(file_path, 'w', encoding='utf-8'),
                              indent=indent, ensure_ascii=ensure_ascii)
        )
    except Exception as e:
        print(f"Error occurred while saving JSON file {file_path}: {str(e)}")

def pil_images_to_base64(img, format="PNG"):
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

async def async_encode_images_batch(img_paths):
    """异步批量编码图片为base64格式"""
    try:
        loop = asyncio.get_event_loop()

        def encode_single_image(img_path):
            with open(img_path, 'rb') as f:
                img_data = f.read()
                return base64.b64encode(img_data).decode('utf-8')

        tasks = [
            loop.run_in_executor(None, encode_single_image, img_path)
            for img_path in img_paths
        ]
        return await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error occurred while batch encoding images: {str(e)}")
        return []

def load_converter(scale=DEFAULT_SCALE) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter