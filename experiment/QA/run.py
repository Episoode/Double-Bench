import json
import base64
import os
import random
import asyncio
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter
import statistics
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import functools


class UnifiedAPIProcessor:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.image_cache = {}
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    def get_language_name(self, lang_code: str) -> str:
        lang_map = {
            'ar': 'Arabic',
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'ja': 'Japanese',
            'zh': 'Chinese'
        }
        return lang_map.get(lang_code, 'English')

    def find_image_path(self, doc_path: str, page_num: int) -> Optional[str]:
        cache_key = f"{doc_path}_{page_num}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]

        page_str = f"{page_num:03d}"
        doc_path_obj = Path(doc_path)

        for ext in self.supported_extensions:
            image_path = doc_path_obj / f"{page_str}{ext}"
            if image_path.is_file():
                result = str(image_path)
                self.image_cache[cache_key] = result
                return result

        self.image_cache[cache_key] = None
        return None

    @functools.lru_cache(maxsize=1000)
    def _encode_image_cached(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            return base64_encoded
        except Exception:
            return None

    async def encode_image_to_base64_async(self, image_path: str) -> Optional[str]:
        try:
            cached_result = self._encode_image_cached(image_path)
            if cached_result:
                return cached_result

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self._encode_image_sync,
                image_path
            )
            return result
        except Exception:
            return None

    def _encode_image_sync(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception:
            return None

    def select_evidence_pages_multi(self, steps: List[Dict[str, Any]]) -> List[int]:
        if not steps:
            return []

        step_count = len(steps)
        evidence_pages = []

        if step_count == 1:
            ref_pages = steps[0].get("reference_page", [])
            evidence_pages = random.sample(ref_pages, min(5, len(ref_pages))) if ref_pages else []
        elif step_count == 2:
            step1, step2 = steps[0], steps[1]
            ref_pages1 = step1.get("reference_page", [])
            ref_pages2 = step2.get("reference_page", [])

            if ref_pages1 and ref_pages2:
                if len(ref_pages1) >= 2 and random.choice([True, False]):
                    evidence_pages.extend(random.sample(ref_pages1, min(3, len(ref_pages1))))
                    evidence_pages.extend(random.sample(ref_pages2, min(2, len(ref_pages2))))
                else:
                    evidence_pages.extend(random.sample(ref_pages1, min(2, len(ref_pages1))))
                    evidence_pages.extend(random.sample(ref_pages2, min(3, len(ref_pages2))))
        elif step_count in [3, 4]:
            for step in steps:
                ref_pages = step.get("reference_page", [])
                if ref_pages:
                    evidence_pages.append(random.choice(ref_pages))


        return list(dict.fromkeys(evidence_pages))[:4]  # deduplicate and keep order

    def select_evidence_pages_single(self, reference_pages: List[int]) -> List[int]:
        if not reference_pages:
            return []
        return random.sample(reference_pages, min(3, len(reference_pages)))

    def create_multimodal_messages(self, question: str, language_code: str, base64_images: List[str]) -> List[
        Dict[str, Any]]:
        language_name = self.get_language_name(language_code)

        user_content = [{"type": "text", "text": f"Question: {question}"}]
        user_content.extend([
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "low"
                }
            } for base64_img in base64_images
        ])

        return [
            {
                "role": "system",
                "content": f"""
            ## Role
            You are an expert analyst and helpful assistant.

            ## Task
            Your primary task is to thoroughly analyze the user-provided image(s) to answer their question. Your answer must be
            supported by the visual evidence and details found within the image(s）

            ## Instruction
            1. First, formulate a brief reasoning process. This should explain how you derived your answer from the visual evidence and any contextual knowledge you applied.
            2. Second, provide a concise and direct final answer in {language_name}.

            ## Format
            Your entire response MUST be a single, valid JSON object. This JSON object must contain exactly two keys:

            1. reason: A string containing your English reasoning process.
            2. answer: A string containing your final answer in the requested language."""
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

    def create_text_only_messages(self, question: str, language_code: str) -> List[Dict[str, Any]]:
        """Create text-only OpenAI API messages"""
        language_name = self.get_language_name(language_code)

        return [
            {
                "role": "system",
                "content": "You are an expert assistant. Your task is to provide a concise answer to the user's question. "
                           "You must follow these instructions strictly:\n"
                           "1. First, think about the question to form a brief reasoning process. This reasoning must be in English.\n"
                           "2. Second, provide a concise, direct answer to the question in the requested language.\n"
                           "3. Your final output MUST be a valid JSON object with exactly two keys: \"reason\" and \"answer\"."
            },
            {
                "role": "user",
                "content": f"Please answer the following question in {language_name}.\nQuestion: \"{question}\""
            }
        ]

    async def call_openai_api(self, messages: List[Dict[str, Any]], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        max_retries = 3
        backoff_factor = 1.5

        for attempt in range(max_retries):
            async with semaphore:
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        response_format={"type": "json_object"},
                        max_tokens=500,
                        temperature=0.5
                    )

                    llm_output_str = response.choices[0].message.content
                    llm_output_json = json.loads(llm_output_str)

                    generated_answer = llm_output_json.get("answer", "Error: 'answer' key not found in LLM response.")
                    reason = llm_output_json.get("reason", "No reasoning provided.")

                    return {
                        "success": True,
                        "generated_answer": generated_answer,
                        "reason": reason,
                        "error": None
                    }

                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff_factor ** attempt)
                        continue
                    else:
                        return {
                            "success": False,
                            "generated_answer": "Error: Failed to parse JSON response",
                            "reason": "JSON parsing failed",
                            "error": f"JSONDecodeError: {e}"
                        }

                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff_factor ** attempt)
                        continue
                    else:
                        return {
                            "success": False,
                            "generated_answer": f"Error: API call failed. Details: {str(e)}",
                            "reason": "API call failed",
                            "error": str(e)
                        }

        return {
            "success": False,
            "generated_answer": "Error: Max retries exceeded",
            "reason": "Max retries exceeded",
            "error": "Max retries exceeded"
        }

    async def process_images_batch(self, evidence_pages: List[int], doc_path: str) -> tuple[List[str], Dict[str, Any]]:
        base64_images = []
        stats = {
            "found_images": [],
            "missing_pages": [],
            "found_image_count": 0,
            "missing_image_count": 0,
            "total_file_size_bytes": 0,
            "processing_errors": []
        }

        # Process images concurrently
        tasks = []
        for page_num in evidence_pages:
            task = self.process_single_image(page_num, doc_path)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for page_num, result in zip(evidence_pages, results):
            if isinstance(result, Exception):
                stats["processing_errors"].append(f"Error processing page {page_num}: {str(result)}")
                stats["missing_pages"].append(page_num)
                stats["missing_image_count"] += 1
            elif result is None:
                stats["processing_errors"].append(f"Image file not found for page {page_num}")
                stats["missing_pages"].append(page_num)
                stats["missing_image_count"] += 1
            else:
                image_path, base64_encoded, file_size = result
                stats["found_images"].append({
                    "page_num": page_num,
                    "image_path": image_path,
                    "file_size_bytes": file_size,
                    "image_format": Path(image_path).suffix.lower()
                })
                base64_images.append(base64_encoded)
                stats["total_file_size_bytes"] += file_size
                stats["found_image_count"] += 1

        return base64_images, stats

    async def process_single_image(self, page_num: int, doc_path: str) -> Optional[tuple]:
        image_path = self.find_image_path(doc_path, page_num)
        if not image_path:
            return None

        base64_encoded = await self.encode_image_to_base64_async(image_path)
        if not base64_encoded:
            return None

        file_size = os.path.getsize(image_path)
        return image_path, base64_encoded, file_size

    async def process_single_item(self, item: Dict[str, Any], semaphore: asyncio.Semaphore,
                                  mode: str, data_type: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        question = item.get("question", "")
        language = item.get("language", "")
        doc_path = item.get("doc_path", "")
        uid = item.get("uid", "unknown")

        result_item = item.copy()
        result_item["generated_answer"] = ""

        stats_info = {
            "uid": uid,
            "processing_success": False,
            "selected_evidence_pages": [],
            "found_images": [],
            "missing_pages": [],
            "found_image_count": 0,
            "missing_image_count": 0,
            "total_file_size_bytes": 0,
            "reasoning": "",
            "processing_errors": [],
            "mode": mode,
            "data_type": data_type
        }

        if not question or not language:
            stats_info["processing_errors"].append("Missing required fields")
            result_item["generated_answer"] = "Error: Missing required fields"
            return result_item, stats_info

        if mode == "norag":
            messages = self.create_text_only_messages(question, language)
            api_result = await self.call_openai_api(messages, semaphore)

            result_item["generated_answer"] = api_result["generated_answer"]
            stats_info["reasoning"] = api_result["reason"]
            stats_info["processing_success"] = api_result["success"]

            if not api_result["success"]:
                stats_info["processing_errors"].append(f"API Error: {api_result['error']}")

            return result_item, stats_info

        if not doc_path:
            stats_info["processing_errors"].append("Missing doc_path for RAG mode")
            result_item["generated_answer"] = "Error: Missing doc_path for RAG mode"
            return result_item, stats_info

        if data_type == "multi":
            steps = item.get("steps", [])
            if not steps:
                stats_info["processing_errors"].append("No steps found")
                result_item["generated_answer"] = "Error: No steps found"
                return result_item, stats_info
            evidence_pages = self.select_evidence_pages_multi(steps)
        else:  # single
            reference_pages = item.get("reference_page", [])
            if not reference_pages:
                stats_info["processing_errors"].append("No reference_page found")
                result_item["generated_answer"] = "Error: No reference_page found"
                return result_item, stats_info
            evidence_pages = self.select_evidence_pages_single(reference_pages)

        stats_info["selected_evidence_pages"] = evidence_pages

        if not evidence_pages:
            stats_info["processing_errors"].append("No evidence pages selected")
            result_item["generated_answer"] = "Error: No evidence pages selected"
            return result_item, stats_info

        # Process images in batch
        base64_images, image_stats = await self.process_images_batch(evidence_pages, doc_path)

        # Update stats
        stats_info.update(image_stats)

        # If at least one image found, call API
        if stats_info["found_image_count"] > 0:
            messages = self.create_multimodal_messages(question, language, base64_images)
            api_result = await self.call_openai_api(messages, semaphore)

            result_item["generated_answer"] = api_result["generated_answer"]
            stats_info["reasoning"] = api_result["reason"]
            stats_info["processing_success"] = api_result["success"]

            if not api_result["success"]:
                stats_info["processing_errors"].append(f"API Error: {api_result['error']}")
        else:
            result_item["generated_answer"] = "Error: No valid images found"
            stats_info["processing_errors"].append("No valid images found")

        return result_item, stats_info

    async def write_partial_results(self, results: List[Dict[str, Any]], output_path: str,
                                    batch_num: int, is_final: bool = False):
        """Write results in batches to JSON file"""
        try:
            output_file = Path(output_path)

            if batch_num == 1:
                # First batch, create new file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    content = json.dumps(results, indent=2, ensure_ascii=False)
                    await f.write(content)
                print(f"  Batch {batch_num} results written ({len(results)} items)")
            else:
                # Subsequent batches, read existing file and append
                async with aiofiles.open(output_path, 'r', encoding='utf-8') as f:
                    existing_content = await f.read()
                    existing_results = json.loads(existing_content)

                # Merge results
                existing_results.extend(results)

                # Write back to file
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    content = json.dumps(existing_results, indent=2, ensure_ascii=False)
                    await f.write(content)
                print(f"  Batch {batch_num} results appended (+{len(results)} items, total {len(existing_results)})")

            if is_final:
                print(f"  All results saved to: {output_path}")

        except Exception as e:
            print(f"  Error writing batch {batch_num}: {str(e)}")

    async def process_all_requests(self, input_json_path: str, output_json_path: str,
                                   mode: str, data_type: str, max_concurrency: int = 16,
                                   write_batch_size: int = 16) -> Dict[str, Any]:
        """Process all requests, write results in batches, return statistics"""

        # Async file reading
        async with aiofiles.open(input_json_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            data_list = json.loads(content)

        print(f"Start processing {len(data_list)} requests...")
        print(f"Model: {self.model_name}")
        print(f"Max concurrency: {max_concurrency}")
        print(f"Mode: {mode.upper()}")
        print(f"Data type: {data_type.upper()}")
        print(f"Batch write size: {write_batch_size}")

        semaphore = asyncio.Semaphore(max_concurrency)
        all_stats_info = []
        batch_count = 0

        # Group by batch size
        for i in range(0, len(data_list), write_batch_size):
            batch_count += 1
            batch = data_list[i:i + write_batch_size]
            batch_desc = f"Batch {batch_count}/{(len(data_list) + write_batch_size - 1) // write_batch_size}"

            print(f"\nProcessing {batch_desc} (items {i + 1}-{min(i + write_batch_size, len(data_list))})")

            # Process current batch
            tasks = [self.process_single_item(item, semaphore, mode, data_type) for item in batch]

            batch_results = await tqdm.gather(
                *tasks,
                desc=f"Processing {batch_desc} ({mode}-{data_type})",
                unit="item"
            )

            # Separate results and stats
            batch_processed_results = []
            for result_item, stats_info in batch_results:
                batch_processed_results.append(result_item)
                all_stats_info.append(stats_info)

            # Write current batch results
            is_final_batch = (i + write_batch_size >= len(data_list))
            await self.write_partial_results(
                batch_processed_results,
                output_json_path,
                batch_count,
                is_final_batch
            )

        # Generate statistics
        stats_summary = self.analyze_results(all_stats_info, mode, data_type)

        # Async save statistics
        stats_file = output_json_path.replace('.json', '_statistics.json')
        async with aiofiles.open(stats_file, 'w', encoding='utf-8') as f:
            content = json.dumps(stats_summary, ensure_ascii=False, indent=2)
            await f.write(content)

        print(f"\nStatistics saved to: {stats_file}")

        return stats_summary

    def analyze_results(self, all_stats_info: List[Dict[str, Any]], mode: str, data_type: str) -> Dict[str, Any]:
        """Analyze results and compute statistics"""

        total_requests = len(all_stats_info)
        successful_requests = sum(1 for info in all_stats_info if info.get("processing_success", False))
        failed_requests = total_requests - successful_requests

        # Collect error uids
        error_uids = [
            {
                "uid": info.get("uid", "unknown"),
                "errors": info.get("processing_errors", [])
            }
            for info in all_stats_info
            if not info.get("processing_success", False) or info.get("processing_errors")
        ]

        result = {
            "model_name": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_mode": mode,
            "data_type": data_type,
            "analysis_summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate_percent": round((successful_requests / total_requests * 100),
                                              2) if total_requests > 0 else 0
            },
            "error_details": {
                "error_count": len(error_uids),
                "error_uids": error_uids
            },
            "detailed_processing_info": all_stats_info
        }

        # Image statistics (only RAG mode)
        if mode == "rag":
            successful_infos = [info for info in all_stats_info if info.get("processing_success", False)]
            image_counts = [info.get("found_image_count", 0) for info in successful_infos]
            total_file_size = sum(info.get("total_file_size_bytes", 0) for info in all_stats_info)

            # Image format statistics
            image_formats = Counter()
            for info in all_stats_info:
                for img in info.get("found_images", []):
                    fmt = img.get("image_format", "unknown")
                    image_formats[fmt] += 1

            if image_counts:
                result["image_statistics"] = {
                    "avg_images_per_request": round(statistics.mean(image_counts), 3),
                    "median_images_per_request": statistics.median(image_counts),
                    "min_images_per_request": min(image_counts),
                    "max_images_per_request": max(image_counts),
                    "image_count_distribution": dict(Counter(image_counts)),
                    "image_format_distribution": dict(image_formats),
                    "total_file_size_mb": round(total_file_size / (1024 * 1024), 2)
                }

        return result

    def print_statistics(self, stats: Dict[str, Any]):
        """Print statistics report"""
        if not stats:
            return

        print("\n" + "=" * 80)
        print("Unified API Processing Statistics Report (Batch Write Optimized)")
        print("=" * 80)
        print(f"Model: {stats['model_name']}")
        print(f"Processing Time: {stats['timestamp']}")
        print(f"Processing Mode: {stats['processing_mode'].upper()}")
        print(f"Data Type: {stats['data_type'].upper()}")

        # Request statistics
        summary = stats['analysis_summary']
        print(f"\nRequest statistics:")
        print(f"  Total requests: {summary['total_requests']:,}")
        print(f"  Successful requests: {summary['successful_requests']:,}")
        print(f"  Failed requests: {summary['failed_requests']:,}")
        print(f"  Success rate: {summary['success_rate_percent']}%")

        # Error statistics
        error_details = stats['error_details']
        print(f"\nError statistics:")
        print(f"  Requests with errors: {error_details['error_count']}")
        if error_details['error_count'] > 0:
            print(f"  First 5 error UIDs:")
            for error_info in error_details['error_uids'][:5]:
                print(f"    - {error_info['uid']}: {error_info['errors']}")

        # Image statistics (only RAG mode)
        if stats['processing_mode'] == 'rag' and 'image_statistics' in stats:
            img_stats = stats['image_statistics']
            print(f"\nImage statistics:")
            print(f"  Average images per request: {img_stats['avg_images_per_request']}")
            print(f"  Median images per request: {img_stats['median_images_per_request']}")
            print(f"  Min images per request: {img_stats['min_images_per_request']}")
            print(f"  Max images per request: {img_stats['max_images_per_request']}")
            print(f"  Total file size: {img_stats['total_file_size_mb']} MB")

            print(f"\n  Image count distribution:")
            if summary['successful_requests'] > 0:
                for count, frequency in sorted(img_stats['image_count_distribution'].items()):
                    percentage = (frequency / summary['successful_requests']) * 100
                    print(f"    {count} image(s): {frequency:,} requests ({percentage:.1f}%)")

            print(f"\n  Image format distribution:")
            for fmt, count in sorted(img_stats['image_format_distribution'].items()):
                print(f"    {fmt}: {count:,} image(s)")

        print("=" * 80)

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


def main():
    """Main function, controlled by CLI arguments"""
    parser = argparse.ArgumentParser(
        description='Unified API processor supporting RAG/NoRAG and Single/Multi modes (Batch write optimized)')

    # Required arguments
    parser.add_argument('--mode', choices=['rag', 'norag'], required=True,
                        help='Mode: rag (with images) or norag (text only)')
    parser.add_argument('--data-type', choices=['single', 'multi'], required=True,
                        help='Data type: single (reference_page) or multi (steps)')
    parser.add_argument('--input', required=True,
                        help='Input JSON file path (e.g. /path/to/your/input.json)')
    parser.add_argument('--output', required=True,
                        help='Output JSON file path (e.g. /path/to/your/output.json)')

    # Optional arguments
    parser.add_argument('--api-key', required=True,
                        help='OpenAI API key')
    parser.add_argument('--base-url', default="https://api.openai.com/v1",
                        help='API base URL')
    parser.add_argument('--model', default="gpt-4o-mini",
                        help='Model name')
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Max concurrency (default 16)')
    parser.add_argument('--write-batch-size', type=int, default=16,
                        help='Batch write size (default 16)')

    args = parser.parse_args()

    args.base_url = args.base_url.strip()

    print(f"Starting Unified API Processor (Batch Write Optimized)")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Data type: {args.data_type.upper()}")
    print(f"Batch write: {args.write_batch_size} items per write")

    async def run_processor():
        processor = UnifiedAPIProcessor(args.api_key, args.base_url, args.model)

        try:
            stats = await processor.process_all_requests(
                args.input,
                args.output,
                args.mode,
                args.data_type,
                args.concurrency,
                args.write_batch_size
            )
            processor.print_statistics(stats)
        finally:
            if hasattr(processor, 'thread_pool'):
                processor.thread_pool.shutdown(wait=True)

    asyncio.run(run_processor())


if __name__ == "__main__":
    main()