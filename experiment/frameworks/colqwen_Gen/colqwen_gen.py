import os
import json
import argparse
import logging
import asyncio
import base64
import random
from io import BytesIO
from typing import List, Dict, Any

from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm as anext_tqdm

# --- Config ---
MAX_CONCURRENCY = 16

# --- Logging config ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "evaluation_run.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, mode='w')
    ]
)
logger = logging.getLogger(__name__)

# --- Remove sensitive info ---
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")

if not API_KEY:
    raise ValueError("API key is not set. Please set environment variable OPENAI_API_KEY")

if not MODEL_NAME:
    raise ValueError("Model name is not set. Please set environment variable OPENAI_MODEL_NAME")

# --- Prompt definition (unchanged) ---
ANSWER_GENERATION_SYSTEM_PROMPT = """You are an expert document-based question answering system. Your task is to provide accurate, well-structured, and comprehensive answers based solely on the provided document content.

Key Requirements:
1. Content-Based Response
   - Base your answer EXCLUSIVELY on the provided document content
   - Do not incorporate external knowledge or assumptions
   - If you are not exactly sure about the answer, use the answer you have most confidence on, but the answer must be based on the content
   - If the content does not contain any likely answer, explicitly state: "The provided content does not contain enough information to answer this question"

2. Answer Quality
   - Be precise, concise, and directly address the question
   - Structure your response in a clear, logical manner
   - Use bullet points or numbered lists when appropriate
   - Maintain academic rigor and professional tone

3. Content Handling
   - Consider all provided content sections equally
   - If content appears contradictory, try to identify the most reliable source as the basis for your answer

4. Response Format
   - Return with strict json format. You output will be directly parsed, so do not add any other text that hinders the parsing process.
   - Begin with a "thought" key. Collect relavant information here, together with your reasoning process if nesessary.
   - After completing the thought, add a "final_answer" key. This is the final answer to the question.
   - Do not include any other keys or information in the response.

5. Example
   - Assume the given image is a document with a graph about the sales of a product over time. The question is "What was the sales trend in Q1 2023?"
   Your answer should be:
   {
       "thought": "The document contains a graph showing the sales trend over time. In Q1 2023, the sales were steadily increasing.",
       "final_answer": "The sales trend in Q1 2023 was steadily increasing."
   }
"""

# --- Core functions (evaluation removed, only answer generation) ---
def encode_image_to_base64(image_path: str) -> str:
    """Read image file and encode as base64 string."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        raise

async def generate_answer(
        client: AsyncOpenAI,
        question: str,
        image_paths: List[str]
) -> Dict[str, Any]:
    """Asynchronously call model to generate answer."""
    content_parts = [{"type": "text", "text": question}]
    for path in image_paths:
        try:
            base64_image = encode_image_to_base64(path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        except Exception as e:
            return {"status": "error", "error_message": f"Failed to process image {path}: {e}"}

    messages = [
        {"role": "system", "content": ANSWER_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": content_parts}
    ]
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        response_text = response.choices[0].message.content
        response_json = json.loads(response_text)
        return {"status": "success", "answer": response_json.get("final_answer", response_text)}
    except Exception as e:
        logger.error(f"Error during answer generation for question '{question[:50]}...': {e}")
        return {"status": "error", "error_message": str(e)}

async def process_sample(client: AsyncOpenAI, sample: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Process a single sample asynchronously (only answer generation)"""
    async with semaphore:
        uid = sample.get("uid", "unknown")
        logger.info(f"Processing sample: {uid}")

        question = sample.get("question")
        image_paths = sample.get("retrieval_pages", [])[:5]
        if not all([question, image_paths]):
            logger.warning(f"Skipping sample {uid} due to missing data.")
            return {"uid": uid, "status": "skipped", "error_message": "Missing question or retrieval_pages"}

        gen_result = await generate_answer(client, question, image_paths)
        if gen_result["status"] == "error":
            return {"uid": uid, "status": "generation_failed", **gen_result}

        logger.info(f"Successfully processed sample: {uid}")
        return {
            "uid": uid,
            "status": "success",
            "question": question,
            "generated_answer": gen_result["answer"],
        }

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Only count successful and failed generations."""
    total_samples = len(results)
    successful = [r for r in results if r.get("status") == "success"]
    return {
        "total_samples": total_samples,
        "successful_generations": len(successful),
        "failed_or_skipped": total_samples - len(successful)
    }

# --- Main function (only answer generation) ---
async def main():
    parser = argparse.ArgumentParser(description="Generate answers for questions based on images.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("output_file", type=str, help="Path to save the output JSON file.")
    args = parser.parse_args()

    logger.info("========================================")
    logger.info(f"Starting new generation run.")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Max concurrency: {MAX_CONCURRENCY}")
    logger.info("========================================")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            samples = random.sample(samples, min(len(samples), 100))
        if not isinstance(samples, list): raise TypeError("Input JSON must be a list of objects.")
    except Exception as e:
        logger.error(f"Failed to load or parse input file {args.input_file}: {e}")
        print(f"Error: Failed to load or parse input file {args.input_file}. See logs/evaluation_run.log for details.")
        return

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL) if BASE_URL else AsyncOpenAI(api_key=API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [process_sample(client, sample, semaphore) for sample in samples]
    results = []

    for future in anext_tqdm.as_completed(tasks, desc="Generating"):
        result = await future
        results.append(result)

    statistics = calculate_statistics(results)
    output_data = {
        "settings": {"model": MODEL_NAME, "max_concurrency": MAX_CONCURRENCY},
        "statistics": statistics,
        "detailed_results": results
    }

    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved results to {args.output_file}")
        logger.info(f"Final Statistics: {json.dumps(statistics)}")
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        print(f"Error: Failed to save output file. See logs/evaluation_run.log for details.")

if __name__ == "__main__":
    asyncio.run(main())