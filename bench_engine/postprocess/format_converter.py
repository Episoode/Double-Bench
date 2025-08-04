import json
from typing import List, Dict, Any
from collections import defaultdict


class FormatConverter:
    """
    Multi-hop QA format converter for converting between multi-hop QA format and single question format.

    This class provides functionality to split multi-hop questions into individual single questions
    and merge them back with updated reference pages and metadata.
    """

    def __init__(self):
        """
        Initialize the format converter.
        """
        pass

    def convert(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert multi-hop QA data to single question format.

        Args:
            data: Original multi-hop QA data list

        Returns:
            List of converted single questions
        """
        single_questions = []

        for item in data:
            base_uid = item.get("uid", "")
            if not base_uid:
                print(f"Warning: Found question item missing uid")
                continue

            file_name = item.get("file_name", "")
            modality = item.get("modality", ["text"])
            if isinstance(modality, list) and len(modality) > 0:
                modality_str = modality[0]
            elif isinstance(modality, str):
                modality_str = modality
            else:
                modality_str = "text"

            steps = item.get("steps", [])

            for step_idx, step in enumerate(steps):
                sub_uid = f"{base_uid}_{step_idx}"

                single_question = {
                    "uid": sub_uid,
                    "question": step.get(f"question{step_idx}", ""),
                    "answer": step.get(f"answer{step_idx}", ""),
                    "modality": modality_str,
                    "file_name": file_name,
                    "evidence_page": step.get("reference_page", "")
                }

                single_questions.append(single_question)

        return single_questions

    def merge_questions(self,
                        single_questions: List[Dict[str, Any]],
                        original_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge split single questions back to original multi-hop QA format with updated reference_page.

        Args:
            single_questions: Processed single questions list (with updated reference_page)
            original_data: Original multi-hop QA data

        Returns:
            Merged multi-hop QA data with updated reference_page
        """
        grouped_questions = defaultdict(list)

        for question in single_questions:
            uid = question["uid"]
            if "_" in uid:
                base_uid = uid.split("_")[0]
                step_idx = int(uid.split("_")[1])
                grouped_questions[base_uid].append((step_idx, question))
            else:
                print(f"Warning: Found uid with unexpected format: {uid}")
                continue

        uid_to_index = {}
        for idx, item in enumerate(original_data):
            item_uid = item.get("uid", "")
            if item_uid:
                uid_to_index[item_uid] = idx

        merged_data = []

        for original_item in original_data:
            item_uid = original_item.get("uid", "")
            if not item_uid:
                print(f"Warning: Found item missing uid in original data")
                merged_data.append(original_item.copy())
                continue

            merged_item = original_item.copy()

            if item_uid in grouped_questions:
                sorted_questions = sorted(grouped_questions[item_uid], key=lambda x: x[0])

                all_reference_pages = set()

                original_ref_pages = merged_item.get("reference_page", [])
                if isinstance(original_ref_pages, list):
                    all_reference_pages.update(original_ref_pages)
                elif isinstance(original_ref_pages, (str, int)):
                    all_reference_pages.add(original_ref_pages)

                if "steps" in merged_item:
                    for step_idx, (_, single_question) in enumerate(sorted_questions):
                        if step_idx < len(merged_item["steps"]):
                            step_ref_page = single_question.get("reference_page", [])

                            merged_item["steps"][step_idx]["reference_page"] = step_ref_page

                            if isinstance(step_ref_page, list):
                                all_reference_pages.update(step_ref_page)
                            elif isinstance(step_ref_page, (str, int)) and step_ref_page:
                                all_reference_pages.add(step_ref_page)
                        else:
                            print(f"Warning: Step index out of range for uid {item_uid}: {step_idx}")

                try:
                    valid_pages = []
                    for page in all_reference_pages:
                        if page is not None and page != "":
                            try:
                                valid_pages.append(int(page))
                            except (ValueError, TypeError):
                                valid_pages.append(page)

                    merged_item["reference_page"] = sorted(list(set(valid_pages)))

                except Exception as e:
                    print(f"Warning: Error processing reference_page for uid {item_uid}: {e}")
                    merged_item["reference_page"] = list(all_reference_pages)

            else:
                print(f"Warning: No corresponding single questions found for uid {item_uid}")

            merged_data.append(merged_item)

        return merged_data

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from file.

        Args:
            file_path: Input file path

        Returns:
            List of loaded data

        Raises:
            FileNotFoundError: File does not exist
            json.JSONDecodeError: JSON parsing error
            ValueError: Invalid data format
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and "questions" in data:
            data = data["questions"]
        elif not isinstance(data, list):
            raise ValueError("Invalid input data format, expected list or dict with 'questions' key")

        return data

    def save_data(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save data to file.

        Args:
            data: Data to save
            file_path: Output file path

        Raises:
            IOError: File write error
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Process file: load, convert, and save.

        Args:
            input_path: Input file path
            output_path: Output file path

        Returns:
            Whether the operation was successful
        """
        data = self.load_data(input_path)

        converted_data = self.convert(data)

        self.save_data(converted_data, output_path)

        print(f"Conversion completed!")
        print(f"Original question count: {len(data)}")
        print(f"Converted sub-question count: {len(converted_data)}")
        print(f"Results saved to: {output_path}")

        return True

    def process_merge_file(self,
                           processed_single_questions_path: str,
                           original_data_path: str,
                           output_path: str) -> bool:
        """
        Process merge file: load processed single questions and original data, merge and save.

        Args:
            processed_single_questions_path: Path to processed single questions file
            original_data_path: Path to original multi-hop QA data file
            output_path: Output file path

        Returns:
            Whether the operation was successful
        """
        try:
            single_questions = self.load_data(processed_single_questions_path)

            original_data = self.load_data(original_data_path)

            merged_data = self.merge_questions(single_questions, original_data)

            self.save_data(merged_data, output_path)

            print(f"Merge completed!")
            print(f"Processed single question count: {len(single_questions)}")
            print(f"Original multi-hop question count: {len(original_data)}")
            print(f"Merged question count: {len(merged_data)}")
            print(f"Results saved to: {output_path}")

            return True

        except Exception as e:
            print(f"Error occurred during merge process: {str(e)}")
            return False
