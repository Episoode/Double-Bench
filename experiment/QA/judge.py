import os
import json
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from pathlib import Path


def create_scoring_prompt(question: str, ground_truth_answer: str, generated_answer: str) -> list:
    system_prompt = '''
    You are a fair and objective grader. Your judgment should be based on a balanced assessment.

    ##Task Definition:
        Your primary task is to evaluate the "Generated Answer" by comparing it against the "Ground Truth Answer", taking
        into account the original Question. Based on this evaluation, you will assign a single integer score from 1 to 10.

    ##Scoring Rubric: You must adhere to the following 1-10 scale.
        • Score 1-3 (Poor): The answer is largely incorrect, irrelevant, contains significant inaccuracies or hallucinations, or
        demonstrates a fundamental misunderstanding of the question.
        • Score 4-6 (Acceptable): The answer is partially correct but either misses important information, is somewhat vague,
        or contains minor inaccuracies. It shows some understanding but is not comprehensive.
        • Score 7-8 (Good): The answer is correct and aligns well with the ground truth. It covers most key aspects but might
        lack a few minor details, could be slightly less concise, or have some minor phrasing improvements.
        • Score 9-10 (Excellent): The answer is fully correct, complete, and concise. It accurately captures all essential
        aspects of the ground truth and is well-articulated.

    ##Format Instructions:
        Your response MUST be a valid JSON object with exactly two keys:
        • reason: A brief, one-sentence justification for your score.
        • score: An integer from 1 to 10.
    '''

    user_prompt = (
        "Please evaluate the 'Generated Answer' using the strict rubric provided.\n\n"
        f"--- Question ---\n{question}\n\n"
        f"--- Ground Truth Answer ---\n{ground_truth_answer}\n\n"
        f"--- Generated Answer to Evaluate ---\n{generated_answer}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


async def score_single_item(item: dict, client: AsyncOpenAI, model_name: str, semaphore: asyncio.Semaphore) -> dict:
    """
    Process a single JSON object: call the scoring API, parse the result, and update the score in the object.
    """
    question = item.get("question")
    ground_truth_answer = item.get("answer")
    generated_answer = item.get("generated_answer")

    if not all([question, ground_truth_answer, generated_answer]):
        item['score'] = -1  # Use -1 to indicate scoring failure
        return item

    if "Error:" in str(generated_answer):
        item['score'] = 1
        return item

    messages = create_scoring_prompt(question, ground_truth_answer, generated_answer)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            llm_output_str = response.choices[0].message.content
            llm_output_json = json.loads(llm_output_str)

            score = llm_output_json.get("score", -1)
            item['score'] = score

        except Exception as e:
            item['score'] = -1

    return item


async def main(api_key, base_url, model_name, input_path, output_path, max_concurrency):
    """
    Main function: read file, score all items concurrently, then save results.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file: {input_path}")
        return

    tasks = [score_single_item(item, client, model_name, semaphore) for item in data_list]

    scored_results = await tqdm.gather(
        *tasks,
        desc="Scoring",
        unit="item"
    )

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_results, f, indent=2, ensure_ascii=False)

    print(f"\nScoring complete! Results saved to: {output_path}")


# ==============================================================================
# --- Configuration Section ---
# ==============================================================================
if __name__ == "__main__":
    # --- API and model configuration ---
    API_KEY = ""
    BASE_URL = ""
    SCORING_MODEL_NAME = ""
    MAX_CONCURRENCY = 16

    # --- File path configuration ---
    ANSWER_MODEL_NAME = ""
    INPUT_FILE_TO_SCORE = f"{ANSWER_MODEL_NAME}.json"

    SCORED_OUTPUT_FILE = f"{ANSWER_MODEL_NAME}_scored.json"

    # --- Run main program ---
    asyncio.run(main(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=SCORING_MODEL_NAME,
        input_path=INPUT_FILE_TO_SCORE,
        output_path=SCORED_OUTPUT_FILE,
        max_concurrency=MAX_CONCURRENCY
    ))