import asyncio
import os
import time
from typing import Dict, Any, Optional
from openai import AsyncOpenAI

from singlehop_generator import SinglehopQAGenerator
from singlehop_judger import SinglehopQAJudger
from bench_engine.utils import load_openai_client, async_load_json


async def run_generation_stage(client: Optional[AsyncOpenAI] = None) -> Dict[str, Any]:
    """
    Runs the QA pair generation stage.

    Args:
        client: OpenAI async client for API calls

    Returns:
        Dictionary containing generation stage statistics including success status,
        number of generated QA pairs, execution time, and output file path
    """
    print("=" * 60)
    print("Starting QA pair generation stage...")
    print("=" * 60)

    start_time = time.time()

    try:
        generator = SinglehopQAGenerator(client=client)
        output_path = await generator.generate_base_bench()

        generated_data = await async_load_json(output_path)
        if generated_data is None:
            raise Exception(f"Unable to read generated file: {output_path}")

        generation_time = time.time() - start_time

        stats = {
            'stage': 'generation',
            'generated_qa_pairs': len(generated_data),
            'generation_time': generation_time,
            'output_file': output_path,
            'success': True
        }

        print(f"\nGeneration stage completed!")
        print(f"Generated QA pairs count: {len(generated_data)}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Output file: {output_path}")

        return stats

    except Exception as e:
        error_stats = {
            'stage': 'generation',
            'error': str(e),
            'success': False,
            'generation_time': time.time() - start_time
        }
        print(f"Generation stage error: {str(e)}")
        return error_stats


async def run_judgment_stage(client: Optional[AsyncOpenAI] = None) -> Dict[str, Any]:
    """
    Runs the QA pair judgment stage to evaluate quality of generated pairs.

    Args:
        client: OpenAI async client for API calls

    Returns:
        Dictionary containing judgment stage statistics including success status,
        number of kept pairs, and quality metrics
    """
    print("=" * 60)
    print("Starting QA pair judgment stage...")
    print("=" * 60)

    try:
        judger = SinglehopQAJudger(client=client)

        if not os.path.exists(judger.input_json_path):
            error_msg = f"Input file does not exist: {judger.input_json_path}"
            print(error_msg)
            return {
                'stage': 'judgment',
                'error': error_msg,
                'success': False
            }

        stats = await judger.judge_dataset()
        stats['stage'] = 'judgment'
        stats['success'] = True

        return stats

    except Exception as e:
        error_stats = {
            'stage': 'judgment',
            'error': str(e),
            'success': False
        }
        print(f"Judgment stage error: {str(e)}")
        return error_stats


async def run_full_pipeline(client: Optional[AsyncOpenAI] = None) -> Dict[str, Any]:
    """
    Runs the complete QA pair generation and judgment pipeline.

    Args:
        client: OpenAI async client for API calls. If None, a new client will be created.

    Returns:
        Dictionary containing comprehensive statistics from all pipeline stages including
        generation stats, judgment stats, and overall pipeline summary
    """
    print("=" * 80)
    print("Starting complete single-hop QA dataset generation and judgment pipeline")
    print("=" * 80)

    pipeline_start_time = time.time()

    if client is None:
        client = load_openai_client(async_mode=True)

    print("\nStage 1: QA pair generation")
    print("-" * 40)

    generation_stats = await run_generation_stage(client)

    if not generation_stats.get('success', False):
        return {
            'pipeline_status': 'failed_at_generation',
            'generation_stats': generation_stats,
            'total_time': time.time() - pipeline_start_time
        }

    print("\nStage 2: QA pair quality judgment")
    print("-" * 40)

    judgment_stats = await run_judgment_stage(client)

    total_time = time.time() - pipeline_start_time

    pipeline_stats = {
        'pipeline_status': 'completed' if judgment_stats.get('success', False) else 'failed_at_judgment',
        'generation_stats': generation_stats,
        'judgment_stats': judgment_stats,
        'total_time': total_time,
        'pipeline_summary': {
            'initial_qa_pairs': generation_stats.get('generated_qa_pairs', 0),
            'final_qa_pairs': judgment_stats.get('kept_pairs', 0),
            'overall_keep_rate': (
                judgment_stats.get('kept_pairs', 0) / generation_stats.get('generated_qa_pairs', 1)
                if generation_stats.get('generated_qa_pairs', 0) > 0 else 0
            ),
            'total_processing_time': total_time
        }
    }

    print_pipeline_summary(pipeline_stats)

    return pipeline_stats


def print_pipeline_summary(stats: Dict[str, Any]) -> None:
    """
    Prints the final summary of pipeline execution results.

    Args:
        stats: Dictionary containing pipeline statistics including success status,
               generation and judgment results, and performance metrics
    """
    print("\n" + "=" * 80)
    print("Pipeline execution completed - Final summary")
    print("=" * 80)

    if stats['pipeline_status'] == 'completed':
        summary = stats['pipeline_summary']
        judgment_stats = stats.get('judgment_stats', {})

        print(f"Pipeline executed successfully!")
        print(f"Initial generated QA pairs count: {summary['initial_qa_pairs']}")
        print(f"Final kept QA pairs count: {summary['final_qa_pairs']}")
        print(f"Overall keep rate: {summary['overall_keep_rate']:.2%}")
        print(f"Total processing time: {summary['total_processing_time']:.2f} seconds")
        print(f"Final output file: {judgment_stats.get('output_file', 'N/A')}")
    else:
        print(f"Pipeline execution failed: {stats['pipeline_status']}")
        if 'generation_stats' in stats and 'error' in stats['generation_stats']:
            print(f"Generation stage error: {stats['generation_stats']['error']}")
        if 'judgment_stats' in stats and 'error' in stats['judgment_stats']:
            print(f"Judgment stage error: {stats['judgment_stats']['error']}")


async def main():
    """
    Main function that runs the complete QA generation and judgment pipeline.

    Creates a shared OpenAI client and executes the full pipeline, returning
    comprehensive statistics about the execution results.
    """
    client = load_openai_client(async_mode=True)

    print("Launching complete QA generation and judgment pipeline...")
    stats = await run_full_pipeline(client)

    return stats


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())