import networkx as nx
import openai
import os
import random
import json
import asyncio
from tqdm import tqdm
from bench_engine.config import OPENAI_API_KEY, TARGET_DIR, MULTIHOP_QA_JSON_PATH
from bench_engine.prompt import (
    RELATIONSHIP_EVALUATE,
    RELATIONSHIP_SELECT,
    STEP_QA_GENERATE,
    STEP_QA_USER_PROMPT,
    QUESTION_CHAIN,
    QUESTION_CHAIN_USER_PROMPT
)


class MultiHopQAGenerator:
    """
    Multi-hop question-answer generator class for creating complex reasoning questions.

    This class generates multi-hop reasoning questions by traversing knowledge graphs
    and creating question chains that require multiple steps of reasoning to answer.
    """

    def __init__(self, api_key=None, max_concurrent=16, max_additional_hops=2,
                 min_degree_threshold=5, min_neighbor_degree=3, candidate_count=5):
        """
        Initialize the multi-hop QA generator.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY from config
            max_concurrent: Maximum number of concurrent API calls
            max_additional_hops: Maximum number of reasoning hops
            min_degree_threshold: Minimum degree threshold for high-degree nodes
            min_neighbor_degree: Minimum degree for neighbor nodes
            candidate_count: Number of candidate nodes to consider
        """
        self.api_key = api_key if api_key is not None else OPENAI_API_KEY
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.max_concurrent = max_concurrent
        self.max_additional_hops = max_additional_hops
        self.min_degree_threshold = min_degree_threshold
        self.min_neighbor_degree = min_neighbor_degree
        self.candidate_count = candidate_count

    async def get_relation_text(self, edge_data):
        """
        Extract meaningful relationship text from edge data.

        Args:
            edge_data: Edge data dictionary containing relationship information

        Returns:
            str: Processed relationship text for the edge
        """
        description = edge_data.get('description')
        keywords = edge_data.get('keywords')

        if description and description != 'Unknown':
            return description.strip().replace('{', '').replace('}', '').replace("'", "")
        elif keywords and keywords != 'Unknown':
            keywords_list = keywords.strip().replace('[', '').replace(']', '').replace("'", "").split(',')
            return ", ".join([k.strip() for k in keywords_list if k.strip()])
        else:
            return "is connected to"

    async def select_best_next_node(self, current_node, candidate_nodes_with_relations):
        """
        Select the best next node from candidates using LLM evaluation.

        Args:
            current_node: Current node identifier
            candidate_nodes_with_relations: List of candidate nodes with their relationship data

        Returns:
            dict: Selected candidate node information with reasoning
        """
        candidates_json = json.dumps(candidate_nodes_with_relations)

        prompt_messages = [
            {"role": "system", "content": RELATIONSHIP_EVALUATE},
            {"role": "user", "content": RELATIONSHIP_SELECT.format(
                current_node=current_node,
                candidates_json=candidates_json
            )}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt_messages,
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            selected_index = result.get('selected_index')
            reasoning = result.get('reasoning')

            if selected_index is not None and 0 <= selected_index < len(candidate_nodes_with_relations):
                selected = candidate_nodes_with_relations[selected_index]
                selected["reasoning"] = reasoning
                return selected
            else:
                candidate_nodes_with_relations[0]["reasoning"] = "Fallback selection (invalid index)"
                return candidate_nodes_with_relations[0]

        except json.JSONDecodeError:
            candidate_nodes_with_relations[0]["reasoning"] = "Fallback selection (JSON decode error)"
            return candidate_nodes_with_relations[0]
        except Exception as e:
            print(f"Error in select_best_next_node: {e}")
            candidate_nodes_with_relations[0]["reasoning"] = "Fallback selection (API or other error)"
            return candidate_nodes_with_relations[0]

    async def generate_step_qa(self, current_node_id, relation_text, next_node_id):
        """
        Generate a single-hop question-answer pair for one reasoning step.

        Args:
            current_node_id: Current node identifier
            relation_text: Relationship description text
            next_node_id: Next node identifier

        Returns:
            dict: Question-answer pair result or None if generation fails
        """
        prompt_messages = [
            {"role": "system", "content": STEP_QA_GENERATE},
            {"role": "user", "content": STEP_QA_USER_PROMPT.format(
                current_node_id=current_node_id,
                relation_text=relation_text,
                next_node_id=next_node_id
            )}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt_messages,
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            print("JSON decode error for API response (step QA)")
            return None
        except Exception as e:
            print(f"Error in generate_step_qa: {e}")
            return None

    async def chain_questions(self, previous_cumulative_q, new_step_q, entity_to_replace):
        """
        Chain a new step question into the previous cumulative question.

        Args:
            previous_cumulative_q: Previous cumulative question text
            new_step_q: New step question to be chained
            entity_to_replace: Entity name to be replaced in the chaining process

        Returns:
            dict: Chained question result or None if chaining fails
        """
        prompt_messages = [
            {"role": "system", "content": QUESTION_CHAIN},
            {"role": "user", "content": QUESTION_CHAIN_USER_PROMPT.format(
                previous_cumulative_q=previous_cumulative_q,
                new_step_q=new_step_q,
                entity_to_replace=entity_to_replace
            )}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt_messages,
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            print("JSON decode error for API response (question chaining)")
            return None
        except Exception as e:
            print(f"Error calling OpenAI API (question chaining): {e}")
            return None

    async def generate_multi_hop_reasoning_iterative(self, graph_path, doc_name, semaphore):
        """
        Generate multi-hop reasoning questions using iterative graph traversal.

        Args:
            graph_path: Path to the graph file
            doc_name: Document name identifier
            semaphore: Semaphore for controlling concurrency

        Returns:
            dict: Reasoning path result with questions and answers, or None if generation fails
        """
        async with semaphore:
            try:
                graph = nx.read_graphml(graph_path)
            except FileNotFoundError:
                print(f"Error: Graph file not found at {graph_path}")
                return None
            except Exception as e:
                print(f"Error loading graph {graph_path}: {e}")
                return None

            high_degree_nodes = [n for n, degree in graph.degree() if degree > self.min_degree_threshold]
            high_degree_nodes = high_degree_nodes[3:] if len(high_degree_nodes) > 3 else high_degree_nodes

            if not high_degree_nodes or len(high_degree_nodes) < 1:
                return None

            start_node = random.choice(high_degree_nodes)
            current_node = start_node
            path_nodes = [start_node]
            cumulative_q = None
            cumulative_a = None
            reasoning_path = {
                "document_name": doc_name,
                "initial_node": start_node,
                "steps": [],
                "modality": []
            }

            for i in range(self.max_additional_hops + 1):
                suitable_neighbors = []
                for neighbor in graph.neighbors(current_node):
                    if graph.degree(neighbor) > self.min_neighbor_degree and (
                            len(path_nodes) < 2 or neighbor != path_nodes[-2]):
                        suitable_neighbors.append(neighbor)

                if not suitable_neighbors:
                    break

                candidate_count = min(self.candidate_count, len(suitable_neighbors))
                candidate_nodes = random.sample(suitable_neighbors, candidate_count)
                candidate_nodes_with_relations = []
                for next_node_cand in candidate_nodes:
                    edge_data = graph.get_edge_data(current_node, next_node_cand)
                    if edge_data:
                        relation_text = await self.get_relation_text(edge_data)
                        candidate_nodes_with_relations.append({
                            "node_id": next_node_cand,
                            "relation_text": relation_text,
                            "edge_data": edge_data
                        })

                if not candidate_nodes_with_relations:
                    break

                best_candidate = await self.select_best_next_node(current_node, candidate_nodes_with_relations)
                if not best_candidate or "node_id" not in best_candidate:
                    print(f"Unable to select best next node from {current_node} in {doc_name}")
                    break

                next_node = best_candidate["node_id"]
                relation_text = best_candidate["relation_text"]
                edge_data = best_candidate["edge_data"]

                reference_page = edge_data.get('page', '')
                modality = edge_data.get('modality', '')

                if modality and modality not in reasoning_path["modality"]:
                    reasoning_path["modality"].append(modality)

                step_qa_result = await self.generate_step_qa(current_node, relation_text, next_node)
                if not step_qa_result:
                    break

                step_q_initial = step_qa_result.get('question')
                current_step_answer = step_qa_result.get('answer')
                step_data = {
                    "current_node": current_node,
                    "next_node": next_node,
                    "relation_text": relation_text,
                    "question_before_replace": step_q_initial,
                    "answer": current_step_answer,
                    "reference_page": reference_page
                }

                if i == 0:
                    cumulative_q = step_q_initial
                    cumulative_a = current_step_answer
                    step_data["question_after_replace"] = cumulative_q
                    reasoning_path["initial_question"] = cumulative_q
                    reasoning_path["initial_answer"] = cumulative_a
                else:
                    entity_to_replace = path_nodes[-1]
                    step_data["entity_to_replace"] = entity_to_replace
                    step_data["question_for_entity_replace"] = cumulative_q
                    chain_result = await self.chain_questions(cumulative_q, step_q_initial, entity_to_replace)
                    if not chain_result:
                        break
                    cumulative_q = chain_result.get('chained_question')
                    cumulative_a = current_step_answer
                    step_data["question_after_replace"] = cumulative_q

                reasoning_path["steps"].append(step_data)
                path_nodes.append(next_node)
                current_node = next_node

            if len(path_nodes) < 2 or not cumulative_q:
                return None

            reasoning_path["final_question"] = cumulative_q
            reasoning_path["final_answer"] = cumulative_a
            return reasoning_path

    async def process_document_questions(self, doc_name, graphml_path, semaphore, questions_per_doc):
        """
        Process multiple question generation tasks for a single document.

        Args:
            doc_name: Document name identifier
            graphml_path: Path to the graph file
            semaphore: Semaphore for controlling concurrency
            questions_per_doc: Number of questions to generate per document

        Returns:
            list: List of successfully generated question results
        """
        tasks = []
        for _ in range(questions_per_doc):
            tasks.append(self.generate_multi_hop_reasoning_iterative(graphml_path, doc_name, semaphore))

        doc_results_from_gather = await asyncio.gather(*tasks)
        successful_results = [result for result in doc_results_from_gather if result is not None]
        return successful_results

    def collect_valid_documents(self, root_directory):
        """
        Collect valid document paths that contain graph files.

        Args:
            root_directory: Root directory path to search for documents

        Returns:
            list: List of valid documents, each element is (doc_name, graph_path) tuple
        """
        valid_documents = []

        try:
            subdirs = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
            for doc_folder in subdirs:
                doc_path_val = os.path.join(root_directory, doc_folder)
                graphml_path = os.path.join(doc_path_val, 'wkdir', 'graph_chunk_entity_relation.graphml')
                if os.path.exists(graphml_path):
                    valid_documents.append((doc_folder, graphml_path))
        except Exception as e:
            print(f"Error accessing root directory '{root_directory}': {e}")
            return []

        return valid_documents

    def write_questions_to_file(self, output_json_path, all_questions):
        """
        Write questions to JSON file (direct overwrite).

        Args:
            output_json_path: Output file path
            all_questions: List of all questions to write
        """
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(all_questions, f_out, ensure_ascii=False, indent=2)

    async def process_all_documents_async(self, root_directory=None, output_json_path=None, questions_per_doc=10):
        """
        Asynchronously process all documents to generate multi-hop questions.

        Args:
            root_directory: Root directory path. If None, uses TARGET_DIR from config
            output_json_path: Output JSON file path. If None, uses MULTIHOP_QA_JSON_PATH from config
            questions_per_doc: Number of questions to generate per document

        Returns:
            bool: Whether processing was successful
        """
        if root_directory is None:
            root_directory = TARGET_DIR
        if output_json_path is None:
            output_json_path = MULTIHOP_QA_JSON_PATH

        valid_documents = self.collect_valid_documents(root_directory)
        if not valid_documents:
            print("No valid document folders found for processing")
            return False

        print(f"Found {len(valid_documents)} documents to process")

        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        semaphore = asyncio.Semaphore(self.max_concurrent)
        progress = tqdm(total=len(valid_documents), desc="Processing documents")

        all_questions = []
        question_id_counter = 1

        for doc_name, graphml_path in valid_documents:
            doc_specific_results = await self.process_document_questions(
                doc_name, graphml_path, semaphore, questions_per_doc
            )

            if doc_specific_results:
                for question_data in doc_specific_results:
                    question_data["id"] = question_id_counter
                    question_id_counter += 1
                    all_questions.append(question_data)

            progress.update(1)

        progress.close()

        self.write_questions_to_file(output_json_path, all_questions)

        print(f"Successfully processed {len(valid_documents)} documents, generated {len(all_questions)} questions")
        print(f"Results saved to: {output_json_path}")
        return True


async def main():
    """
    Main function to demonstrate multi-hop QA generation process.

    Creates a MultiHopQAGenerator instance and processes all documents
    using configuration from config file to generate multi-hop questions.
    """
    generator = MultiHopQAGenerator()

    await generator.process_all_documents_async(questions_per_doc=15)


if __name__ == "__main__":
    asyncio.run(main())