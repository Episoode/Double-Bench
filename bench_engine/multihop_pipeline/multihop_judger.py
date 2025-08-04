import json
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import List, Dict, Optional, Any

from bench_engine.prompt import MULTIHOP_QA_FILTER_PROMPT
from bench_engine.utils import load_openai_client
from bench_engine.config import MULTIHOP_QA_JSON_PATH, MULTIHOP_JUDGED_JSON_PATH, MAX_CONCURRENT_REQUESTS, \
    QA_JUDGE_MODEL


class MultihopQAJudger:
    """
    Multi-hop question-answer quality evaluator for assessing and filtering multi-hop reasoning questions.

    This class provides functionality to evaluate the quality of multi-hop reasoning questions
    using LLM-based filtering and automatic single-step question removal.
    """

    def __init__(self,
                 model: str = QA_JUDGE_MODEL,
                 max_tokens: int = 500,
                 temperature: float = 0.5):
        """
        Initialize the multi-hop QA evaluator.

        Args:
            model: Model name to use for evaluation
            max_tokens: Maximum number of tokens for model response
            temperature: Temperature parameter for model generation
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.client = load_openai_client(async_mode=True)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    def format_question_to_new_structure(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert question format to new structure.

        Args:
            question: Original question dictionary

        Returns:
            Converted question dictionary in new format
        """
        new_question = {
            "file_name": question.get('document_name', ''),
            "question": question.get('final_question', ''),
            "answer": question.get('final_answer', ''),
            "reference_page": question.get('reference_page', []),
            "modality": question.get('modality', []),
            "steps": []
        }

        steps = question.get('steps', [])
        for i, step in enumerate(steps):
            new_step = {
                f"question{i}": step.get('question_before_replace', ''),
                f"answer{i}": step.get('answer', ''),
                "reference_page": step.get('reference_page', '')
            }
            new_question["steps"].append(new_step)

        return new_question

    def format_questions_to_new_structure(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch convert questions to new structure format and assign uid to each question.

        Args:
            questions: Original question list

        Returns:
            Converted question list in new format, each question contains uid
        """
        new_questions = []
        for i, question in enumerate(questions):
            new_question = self.format_question_to_new_structure(question)
            new_question["uid"] = str(i)
            new_questions.append(new_question)
        return new_questions

    def format_steps_for_prompt(self, steps: List[Dict[str, Any]]) -> str:
        """
        Format step list into readable string for LLM prompt.

        Args:
            steps: List of reasoning steps

        Returns:
            Formatted step description string
        """
        step_strings = []
        for i, step in enumerate(steps):
            step_info = [
                f"Step {i + 1}:",
                f"  Current Node: {step.get('current_node', 'N/A')}",
                f"  Relation: {step.get('relation_text', 'N/A')}",
                f"  Next Node (Answer): {step.get('next_node', 'N/A')}",
                f"  Step Question (Before Chaining): {step.get('question_before_replace', 'N/A')}",
                f"  Step Answer: {step.get('answer', 'N/A')}"
            ]

            if i > 0:
                prev_step_answer = steps[i - 1].get('answer', 'N/A') if i - 1 < len(steps) else 'N/A'
                step_info.extend([
                    f"  (Answer from Step {i} was '{prev_step_answer}')",
                    f"  Cumulative Question Before This Step: {step.get('question_for_entity_replace', 'N/A')}",
                    f"  Cumulative Question After This Step: {step.get('question_after_replace', 'N/A')}",
                    f"  Entity Replaced in This Step: {step.get('entity_to_replace', 'N/A')}"
                ])
            step_strings.append("\n".join(step_info))
        return "\n\n".join(step_strings)

    async def filter_single_question(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter single question using LLM evaluation.

        Args:
            question: Question dictionary to be evaluated

        Returns:
            Original question if it passes evaluation, None otherwise
        """
        async with self.semaphore:
            try:
                steps_description = self.format_steps_for_prompt(question.get('steps', []))

                user_content = MULTIHOP_QA_FILTER_PROMPT.format(
                    initial_question=question.get('initial_question', 'N/A'),
                    initial_answer=question.get('initial_answer', 'N/A'),
                    final_question=question.get('final_question', 'N/A'),
                    final_answer=question.get('final_answer', 'N/A'),
                    steps_description=steps_description
                )

                messages = [
                    {"role": "user", "content": user_content}
                ]

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)

                if 'reason' in result and 'keep' in result and result['keep'] in ['yes', 'no']:
                    return question if result['keep'] == 'yes' else None
                else:
                    print(
                        f"Warning: Invalid JSON format from API, question ID {question.get('id', 'N/A')}. Response: {result}")
                    return None

            except json.JSONDecodeError:
                print(f"Error parsing API response JSON, question ID {question.get('id', 'N/A')}.")
                return None
            except Exception as e:
                print(f"Error calling OpenAI API, question ID {question.get('id', 'N/A')}: {e}")
                return None

    def load_questions(self, input_json_path: str) -> List[Dict[str, Any]]:
        """
        Load question list from JSON file.

        Args:
            input_json_path: Input JSON file path

        Returns:
            List of questions
        """
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)

            if isinstance(questions_data, list):
                return questions_data
            elif isinstance(questions_data, dict) and "questions" in questions_data:
                return questions_data["questions"]
            else:
                print(f"Warning: Input file format unexpected, expected list or dict with 'questions' key")
                return []

        except FileNotFoundError:
            print(f"Error: Input file not found {input_json_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file {input_json_path}")
            return []
        except Exception as e:
            print(f"Error loading questions: {e}")
            return []

    def save_questions(self, questions: List[Dict[str, Any]], output_json_path: str) -> bool:
        """
        Save question list to JSON file.

        Args:
            questions: List of questions to save
            output_json_path: Output JSON file path

        Returns:
            Whether saving was successful
        """
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving filtered questions to JSON file: {e}")
            return False

    def auto_filter_single_step_questions(self, questions: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        """
        Automatically filter single-step questions.

        Args:
            questions: Original question list

        Returns:
            Tuple of (multi-step question list, count of filtered single-step questions)
        """
        multi_step_questions = []
        auto_filtered_count = 0

        for question in questions:
            steps = question.get('steps', [])
            if len(steps) <= 1:
                auto_filtered_count += 1
            else:
                multi_step_questions.append(question)

        return multi_step_questions, auto_filtered_count

    async def filter_questions_async(self,
                                     input_json_path: str = None,
                                     output_json_path: str = None) -> bool:
        """
        Asynchronously filter all questions.

        Args:
            input_json_path: Input JSON file path, defaults to configured path
            output_json_path: Output JSON file path, defaults to configured path

        Returns:
            Whether processing was successful
        """
        if input_json_path is None:
            input_json_path = MULTIHOP_QA_JSON_PATH
        if output_json_path is None:
            output_json_path = MULTIHOP_JUDGED_JSON_PATH

        all_questions = self.load_questions(input_json_path)
        if not all_questions:
            print("No questions found in input JSON file.")
            if self.save_questions([], output_json_path):
                print(f"Created empty output file: {output_json_path}")
            return True

        questions_to_evaluate, auto_filtered_count = self.auto_filter_single_step_questions(all_questions)

        print(f"Auto-filtered {auto_filtered_count} questions with only one step.")

        if not questions_to_evaluate:
            print("All questions were auto-filtered.")
            if self.save_questions([], output_json_path):
                print(f"Created empty output file: {output_json_path}")
            return True

        print(f"Starting to filter {len(all_questions)} questions...")
        print(f"Sending {len(questions_to_evaluate)} questions to LLM for evaluation...")

        tasks = [
            self.filter_single_question(question)
            for question in questions_to_evaluate
        ]

        filtered_questions = []

        for completed_task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Filtering questions"):
            result_question = await completed_task
            if result_question is not None:
                filtered_questions.append(result_question)

        filtered_questions = self.format_questions_to_new_structure(filtered_questions)

        if self.save_questions(filtered_questions, output_json_path):
            print(f"\nFiltering completed. Saved {len(filtered_questions)} questions to {output_json_path}")
            print(f"Total filtered questions: {len(all_questions) - len(filtered_questions)}")
            print(f"  - Auto-filtered (single-step): {auto_filtered_count}")
            print(f"  - LLM filtered: {len(questions_to_evaluate) - len(filtered_questions)}")
            return True
        else:
            return False

    def get_statistics(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistical information about the question list.

        Args:
            questions: List of questions

        Returns:
            Dictionary containing statistical information
        """
        if not questions:
            return {"total_questions": 0}

        stats = {
            "total_questions": len(questions),
            "step_distribution": {},
            "has_filter_reason": 0,
            "average_steps": 0,
            "has_uid": 0
        }

        total_steps = 0
        for question in questions:
            steps = question.get('steps', [])
            if steps and isinstance(steps[0], dict):
                step_count = len(steps)
            else:
                step_count = len(steps)

            total_steps += step_count

            stats["step_distribution"][step_count] = stats["step_distribution"].get(step_count, 0) + 1

            if 'filter_reason' in question:
                stats["has_filter_reason"] += 1

            if 'uid' in question:
                stats["has_uid"] += 1

        if len(questions) > 0:
            stats["average_steps"] = total_steps / len(questions)

        return stats


def main():
    """
    Main function to run the multi-hop QA judging process.

    Creates a MultihopQAJudger instance and runs the asynchronous filtering
    process using paths from configuration.
    """
    judger = MultihopQAJudger()

    asyncio.run(judger.filter_questions_async())


if __name__ == "__main__":
    main()