import os
import json
import argparse
import logging
import datetime
import glob
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
import asyncio
from tqdm import tqdm

# Global constants
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N samples

# Get logger for this module
logger = logging.getLogger(__name__)

def update_global_settings(max_retries: int, checkpoint_interval: int) -> None:
    """Update global settings with new values."""
    global MAX_RETRIES, CHECKPOINT_INTERVAL
    MAX_RETRIES = max_retries
    CHECKPOINT_INTERVAL = checkpoint_interval

def setup_logging(input_file: str) -> None:
    """Setup logging configuration based on input filename."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename based on input filename and timestamp
    input_basename = os.path.basename(input_file)
    input_name = os.path.splitext(input_basename)[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{input_name}_judge_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log the start of evaluation
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info(f"Starting Answer Consistency Judgment at {datetime.datetime.now().isoformat()}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*50)

SYSTEM_PROMPT = """You are a strict judge. You must only respond with 'yes' or 'no'."""

async def judge_answer(client: AsyncOpenAI, model: str, reference: str, generated: str, query: str) -> Dict[str, Any]:
    """Judge the generated answer based on the reference answer."""
    prompt = f"""You are a comprehensive judge evaluating the LLM generated answer based on the reference answer.
    You should first collect similarities and differences between the reference answer and the generated answer.
    Then, give a score from 0 to 10 based on the correctness of the generated answer.
    Do allow the generated answer to include additional information if the correct information in the reference answer is already provided.
    Return your answer in strict json format. You output will be directly parsed, so do not add any other text that hinders the parsing process.

    Example 1:
    Query: "How do educational backgrounds influence congressional vote preferences among registered voters?"
    Reference Answer: "Registered voters with postgraduate degrees favor Democrats by 62%, while those with four-year degrees support Democrats by 53%. Preferences are more divided among those without a college degree."
    Generated Answer: "Registered voters' preference for Democrats increases with higher levels of education: 53% among those with a four-year degree and 62% among those with a postgraduate degree, contrasting with the more divided opinions of those without a college degree."
    Output:
    {{
        "evaluation": "The generated answer accurately contains the ratio of registered voters with postgraduate degrees favoring Democrats, and the ratio of registered voters with four-year degrees supporting Democrats. The answer also mentions divided preferences among those without a college degree, which is consistent with the reference answer.",
        "score": 10
    }}

    Example 2:
    Query: "What specifications differentiate the video output capabilities across different models of Roku devices?"
    Reference Answer: "Roku devices vary in their video output capabilities, with Roku 1 and Roku 2 supporting up to 1080p, Roku LT up to 720p, and Roku 3 and Roku 4 supporting up to 4K Ultra HD."
    Generated Answer: "The video output capabilities differentiate across Roku devices as follows: Roku 1, Roku 2, and Roku LT support video output up to 720p, while Roku 3 and Roku 4 support video output from 1080p to 4K Ultra HD."
    Output:
    {{
        "evaluation": "The generated answer correctly states the video output capabilities of Roku LT, Roku 3, and Roku 4. However, it incorrectly states that Roku 1 and Roku 2 support video output up to 720p instead of 1080p. The answer has 3 correct points and 2 incorrect points.",
        "score": 6
    }}

    Example 3:
    Query: "Which celebration occurs during Allhallowtide and involves honoring the dead?"
    Reference Answer: "Halloween is celebrated during Allhallowtide, which involves various traditions related to honoring the dead."
    Generated Answer: "Christmas."
    Output:
    {{
        "evaluation": "The generated answer incorrectly states 'Christmas' instead of 'Halloween'.",
        "score": 0
    }}

    Input:

    Query: {query}
    Reference answer: {reference}
    Generated answer: {generated}
"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        # Get the response and parse JSON
        judgment_text = response.choices[0].message.content.strip()
        try:
            judgment_json = json.loads(judgment_text)
            return judgment_json
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing judgment JSON: {str(e)}")
            return {
                "error": "Failed to parse judgment JSON",
                "raw_response": judgment_text
            }
        
    except Exception as e:
        logger.error(f"Error in judgment: {str(e)}")
        return {"error": str(e)}

def get_checkpoint_dir(input_file: str) -> str:
    """Create and return the path for checkpoints directory."""
    input_basename = os.path.basename(input_file)
    input_name = os.path.splitext(input_basename)[0]
    checkpoint_dir = os.path.join("output", f"checkpoints_{input_name}_judge")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(checkpoint_dir: str, input_file: str, results: List[Dict], current_index: int) -> str:
    """Save checkpoint in a readable format."""
    try:
        # Calculate statistics safely
        total = len(results)
        successful = 0
        failed = 0
        skipped = 0
        
        for result in results:
            if result is None:
                continue
            judgment = result.get('judgment')
            if judgment == 'error':
                failed += 1
            elif judgment == 'skipped':
                skipped += 1
            elif judgment in ['yes', 'no']:
                successful += 1
        
        checkpoint_data = {
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'input_file': input_file,
                'total_samples': total,
                'current_index': current_index,
                'progress_percentage': (current_index / total * 100) if total > 0 else 0,
                'successful_samples': successful,
                'failed_samples': failed,
                'skipped_samples': skipped
            },
            'results': results
        }
        
        # Save checkpoint with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(checkpoint_dir, f"judge_progress_{timestamp}.json")
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        return checkpoint_file
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        # Try to save a minimal checkpoint with just the essential data
        try:
            minimal_checkpoint = {
                'metadata': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'input_file': input_file,
                    'total_samples': len(results),
                    'current_index': current_index,
                    'error': str(e)
                },
                'results': results
            }
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_file = os.path.join(checkpoint_dir, f"judge_progress_emergency_{timestamp}.json")
            with open(emergency_file, 'w') as f:
                json.dump(minimal_checkpoint, f, indent=2)
            logger.info(f"Emergency checkpoint saved: {emergency_file}")
            return emergency_file
        except Exception as emergency_error:
            logger.error(f"Failed to save emergency checkpoint: {str(emergency_error)}")
            raise

def load_checkpoint(checkpoint_dir: str, input_file: str) -> Tuple[List[Dict], int]:
    """Find and load the most recent checkpoint."""
    try:
        # Find all checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "judge_progress_*.json"))
        if not checkpoint_files:
            return [], 0
            
        # Get the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        # Load checkpoint data
        with open(latest_checkpoint, 'r') as f:
            checkpoint_data = json.load(f)
            
        # Verify this is a checkpoint for the correct input file
        if checkpoint_data.get('metadata', {}).get('input_file') != input_file:
            logger.warning(f"Checkpoint file {latest_checkpoint} is for a different input file. Starting fresh.")
            return [], 0
            
        results = checkpoint_data.get('results', [])
        current_index = checkpoint_data.get('metadata', {}).get('current_index', 0)
        
        # Validate results
        if not isinstance(results, list):
            logger.warning(f"Invalid results format in checkpoint {latest_checkpoint}. Starting fresh.")
            return [], 0
            
        logger.info(f"Loaded checkpoint from {latest_checkpoint}")
        logger.info(f"Progress: {current_index}/{len(results)} samples processed")
        return results, current_index
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return [], 0

async def judge_answer_with_retry(client: AsyncOpenAI, model: str, reference: str, generated: str, query: str, max_retries: Optional[int] = None) -> Dict[str, Any]:
    """Judge the generated answer with retry mechanism."""
    global MAX_RETRIES
    retries = max_retries if max_retries is not None else MAX_RETRIES
    
    for attempt in range(retries):
        try:
            result = await judge_answer(client, model, reference, generated, query)
            if "error" not in result:
                return result
                
            if attempt < retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
                
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}, retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"All {retries} attempts failed. Last error: {str(e)}")
                return {"error": str(e)}
    
    return {"error": "All attempts failed"}

async def process_samples(samples: List[Dict], model: str, input_file: str) -> List[Dict]:
    """Process all samples to evaluate answers."""
    global CHECKPOINT_INTERVAL
    client = AsyncOpenAI(api_key='')   # fill in your api key
    checkpoint_dir = get_checkpoint_dir(input_file)
    
    # Validate input samples
    if not samples or not isinstance(samples, list):
        raise ValueError("Input samples must be a non-empty list")
    
    # Try to load checkpoint
    results, start_index = load_checkpoint(checkpoint_dir, input_file)
    
    # If no checkpoint or invalid checkpoint, initialize results with empty dictionaries
    if not results or len(results) != len(samples):
        logger.info("Initializing new results list")
        results = [{
            'uid': f'unknown_{i}',
            'status': 'pending',
            'reason': 'not processed yet'
        } for i in range(len(samples))]
        start_index = 0
    
    last_successful_checkpoint = None

    try:
        for i in tqdm(range(start_index, len(samples)), desc="Evaluating answers", initial=start_index):
            if i >= len(samples):
                logger.warning(f"Index {i} out of range for samples length {len(samples)}")
                break
                
            sample = samples[i]
            if not isinstance(sample, dict):
                logger.warning(f"Sample at index {i} is not a dictionary, skipping...")
                results[i] = {
                    'uid': f'unknown_{i}',
                    'status': 'skipped',
                    'reason': 'invalid sample format'
                }
                continue
                
            reference = sample.get("reference_answer", '')
            generated = sample.get('final_answer', '')
            query = sample.get('query', '')

            if generated == "I cannot find any information from the given pages.":
                results[i] = {
                    'uid': sample.get('uid', f'unknown_{i}'),
                    'reference_answer': reference,
                    'generated_answer': generated,
                    'question': query,
                    'evaluation': {
        "evaluation": "The generated answer does not provide any relevant information.",
        "score": 0
      },
                    'language': sample.get('language', 'unknown'),
                    'status': 'direct judge'
                }
                continue
            
            if not reference or not generated or not query:
                results[i] = {
                    'uid': sample.get('uid', f'unknown_{i}'),
                    'status': 'skipped',
                    'reason': 'missing reference, generated answer, or query'
                }
                continue
                
            try:
                evaluation_result = await judge_answer_with_retry(client, model, reference, generated, query)
                
                results[i] = {
                    'uid': sample.get('uid', f'unknown_{i}'),
                    'reference_answer': reference,
                    'generated_answer': generated,
                    'question': query,
                    'evaluation': evaluation_result,
                    'language': sample.get('language', 'unknown')
                }
                
                # Save checkpoint periodically, but only if we have a successful judgment
                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    try:
                        # Filter out pending results before saving checkpoint
                        checkpoint_results = [
                            r if r.get('judgment') != 'pending' else {
                                'uid': r.get('uid', f'unknown_{idx}'),
                                'judgment': 'skipped',
                                'reason': 'interrupted processing'
                            }
                            for idx, r in enumerate(results)
                        ]
                        checkpoint_file = save_checkpoint(checkpoint_dir, input_file, checkpoint_results, i + 1)
                        if checkpoint_file:
                            last_successful_checkpoint = checkpoint_file
                    except Exception as e:
                        logger.error(f"Error saving checkpoint at index {i}: {str(e)}")
                        # Continue processing even if checkpoint save fails
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {str(e)}")
                results[i] = {
                    'uid': sample.get('uid', f'unknown_{i}'),
                    'judgment': 'error',
                    'reason': str(e)
                }
                
        # Process any remaining pending results
        for i, result in enumerate(results):
            if result.get('judgment') == 'pending':
                results[i] = {
                    'uid': result.get('uid', f'unknown_{i}'),
                    'judgment': 'skipped',
                    'reason': 'not processed'
                }
                
        # Save final checkpoint
        try:
            final_checkpoint = save_checkpoint(checkpoint_dir, input_file, results, len(samples))
            if final_checkpoint:
                last_successful_checkpoint = final_checkpoint
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {str(e)}")
            if last_successful_checkpoint:
                logger.info(f"Last successful checkpoint was: {last_successful_checkpoint}")
        
        # Clean up old checkpoint files if everything completed successfully
        try:
            for old_checkpoint in glob.glob(os.path.join(checkpoint_dir, "judge_progress_*.json")):
                if old_checkpoint != last_successful_checkpoint:  # Keep the last successful checkpoint
                    try:
                        os.remove(old_checkpoint)
                        logger.debug(f"Cleaned up old checkpoint: {old_checkpoint}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up old checkpoint {old_checkpoint}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error during checkpoint cleanup: {str(e)}")
                
    except Exception as e:
        # Save checkpoint on error
        logger.error(f"Error during processing: {str(e)}")
        if results:
            try:
                # Process any pending results before emergency save
                emergency_results = [
                    r if r.get('judgment') != 'pending' else {
                        'uid': r.get('uid', f'unknown_{idx}'),
                        'judgment': 'skipped',
                        'reason': 'interrupted processing'
                    }
                    for idx, r in enumerate(results)
                ]
                emergency_checkpoint = save_checkpoint(checkpoint_dir, input_file, emergency_results, i)
                if emergency_checkpoint:
                    logger.info(f"Emergency checkpoint saved: {emergency_checkpoint}")
            except Exception as checkpoint_error:
                logger.error(f"Failed to save checkpoint after error: {str(checkpoint_error)}")
                if last_successful_checkpoint:
                    logger.info(f"Last successful checkpoint was: {last_successful_checkpoint}")
        raise
        
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Judge answer consistency using GPT-4o-mini')
    parser.add_argument('--input_file', required=True, help='Input JSON file path (output from run_oracle_eval.py)')
    parser.add_argument('--output_file', required=True, help='Output JSON file path for judgment results')
    parser.add_argument('--model', type=str, help='Model to use for judgment (defaults to DOCRAG_MODEL env var or gpt-4o-mini)')
    parser.add_argument('--max_retries', type=int, default=MAX_RETRIES, help=f'Maximum number of retries for failed judgments (default: {MAX_RETRIES})')
    parser.add_argument('--checkpoint_interval', type=int, default=CHECKPOINT_INTERVAL, help=f'Interval for saving checkpoints (default: {CHECKPOINT_INTERVAL})')
    args = parser.parse_args()

    # Update global settings if specified
    update_global_settings(args.max_retries, args.checkpoint_interval)

    # Setup logging
    setup_logging(args.input_file)
    logger = logging.getLogger(__name__)

    # Get model from args or environment variable
    model = args.model
    logger.info(f"Using model: {model}")
    logger.info(f"Max retries: {MAX_RETRIES}")
    logger.info(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")

    try:
        # Load input data
        with open(args.input_file, 'r') as f:
            data = json.load(f)
            
        #if not isinstance(data, dict):
            #raise ValueError("Input file must contain a dictionary")
            
        #if 'samples' not in data:
            #raise ValueError("Input file must contain a 'samples' key")
            
        #samples = data['samples']

        samples = data

        if not isinstance(samples, list):
            raise ValueError("'samples' must be a list")
            
        if not samples:
            raise ValueError("'samples' list is empty")
            
        logger.info(f"Loaded {len(samples)} samples from {args.input_file}")
        
        # Process samples
        results = asyncio.run(process_samples(samples, model, args.input_file))
        
        # Validate results
        if not results or len(results) != len(samples):
            raise ValueError(f"Results length ({len(results) if results else 0}) doesn't match input samples length ({len(samples)})")
        
        # Calculate statistics
        total = len(results)
        evaluated = len([r for r in results if r and r.get('status') not in ['skipped', 'error']])
        scores = [r.get('evaluation', {}).get('score', 0) for r in results if r and r.get('status') not in ['skipped', 'error']]
        avg_score = sum(scores) / len(scores) if scores else 0
        error_count = len([r for r in results if r and r.get('status') == 'error'])
        skipped_count = len([r for r in results if r and r.get('status') == 'skipped'])
        
        # Calculate score distribution
        score_distribution = {
            '0-3': len([s for s in scores if 0 <= s <= 3]),
            '4-6': len([s for s in scores if 4 <= s <= 6]),
            '7-10': len([s for s in scores if 7 <= s <= 10])
        }
        
        # Calculate language-specific statistics
        language_stats = {}
        for result in results:
            if result and result.get('status') not in ['skipped', 'error']:
                lang = result.get('language', 'unknown')
                if lang not in language_stats:
                    language_stats[lang] = {
                        'total': 0,
                        'scores': [],
                        'avg_score': 0,
                        'score_distribution': {
                            '0-2': 0,
                            '3-5': 0,
                            '6-8': 0,
                            '9-10': 0
                        }
                    }
                language_stats[lang]['total'] += 1
                score = result.get('evaluation', {}).get('score', 0)
                language_stats[lang]['scores'].append(score)
                if 0 <= score <= 2:
                    language_stats[lang]['score_distribution']['0-2'] += 1
                elif 3 <= score <= 5:
                    language_stats[lang]['score_distribution']['3-5'] += 1
                elif 6 <= score <= 8:
                    language_stats[lang]['score_distribution']['6-8'] += 1
                elif 9 <= score <= 10:
                    language_stats[lang]['score_distribution']['9-10'] += 1
        
        # Calculate average scores for each language
        for lang in language_stats:
            stats = language_stats[lang]
            stats['avg_score'] = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            # Remove raw scores list to keep the output clean
            del stats['scores']
        
        # Prepare statistics data
        statistics = {
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'model': model,
                'input_file': args.input_file,
                'output_file': args.output_file,
                'total_samples': total,
                'evaluated_samples': evaluated,
                'success_rate': evaluated / total if total > 0 else 0
            },
            'overall_metrics': {
                'average_score': avg_score,
                'score_distribution': score_distribution,
                'error_count': error_count,
                'skipped_count': skipped_count
            },
            'language_metrics': language_stats
        }
        
        # Save statistics to a separate file
        statistics_file = f"{os.path.splitext(args.output_file)[0]}_statistics.json"
        with open(statistics_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        logger.info(f"Statistics saved to: {statistics_file}")
        
        # Prepare output data
        output_data = {
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'model': model,
                'input_file': args.input_file,
                'statistics_file': statistics_file
            },
            'results': results
        }
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        # Print summary
        logger.info("\nEvaluation Summary:")
        logger.info(f"Total samples: {total}")
        logger.info(f"Evaluated samples: {evaluated}")
        logger.info(f"Average score: {avg_score:.2f}")
        logger.info("\nScore Distribution:")
        logger.info(f"0-2: {score_distribution['0-2']} samples")
        logger.info(f"3-5: {score_distribution['3-5']} samples")
        logger.info(f"6-8: {score_distribution['6-8']} samples")
        logger.info(f"9-10: {score_distribution['9-10']} samples")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Skipped: {skipped_count}")
        
        # Print language-specific statistics
        logger.info("\nLanguage-specific Statistics:")
        for lang, stats in language_stats.items():
            logger.info(f"\n{lang.upper()}:")
            logger.info(f"  Total samples: {stats['total']}")
            logger.info(f"  Average score: {stats['avg_score']:.2f}")
            logger.info("  Score Distribution:")
            logger.info(f"    0-2: {stats['score_distribution']['0-2']} samples")
            logger.info(f"    3-5: {stats['score_distribution']['3-5']} samples")
            logger.info(f"    6-8: {stats['score_distribution']['6-8']} samples")
            logger.info(f"    9-10: {stats['score_distribution']['9-10']} samples")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Basic logging setup for fatal errors
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.error(f"Fatal error: {str(e)}")
        raise
