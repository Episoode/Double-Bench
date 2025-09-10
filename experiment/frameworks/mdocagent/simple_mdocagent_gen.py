import json
from openai import OpenAI # Correct import for V1.x SDK
from PIL import Image
import concurrent.futures
import base64
from io import BytesIO
import os # For dummy file creation in example
import requests
from PIL import Image
from openai import OpenAI
import prompts



openai_client = OpenAI(api_key='')  # fill in your api key


def _encode_image_to_base64(image: Image.Image) -> str: # Unchanged

    import base64; from io import BytesIO

    buffered = BytesIO(); image.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode('utf-8')



def invoke_gpt4o(image_paths, prompt):

    text_content = [{"type": "text", "text": prompt}]

    if image_paths is not None:

        for i in image_paths: 

            text_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image_to_base64(Image.open(i))}"}})

    messages = [{"role": "user", "content": text_content}]

    for attempt in range(3):

        try:

            response = openai_client.chat.completions.create(

                model="gpt-4o-mini",

                messages=messages,

                max_tokens=1024,

                temperature=0.1

            )

            return response.choices[0].message.content.strip()

        except Exception as e:

            if attempt == 2:

                print('Error invoking GPT-4o:', e)

                return "Error"

    return "Error"



def general_agent(query, image_paths, text_data):

    prompt = prompts.general_prompt

    prompt += '\n'

    prompt += f'Question: {query}\n'

    prompt += f'Text information: {text_data}\n'

    return invoke_gpt4o(image_paths, prompt)



def critical_agent(query, image_paths, text_data, answer_from_general_agent):

    prompt = prompts.critical_prompt

    prompt += '\n'

    prompt += f'Question: {query}\n'

    prompt += f'General answer from another agent: {answer_from_general_agent}\n'

    prompt += f'Text information: {text_data}\n'

    response = invoke_gpt4o(image_paths, prompt)

    if response.startswith('{') and response.endswith('}'):

        try:

            response_dict = json.loads(response)

            if 'text' in response_dict and 'image' in response_dict:

                return response_dict

        except json.JSONDecodeError:

            pass

    elif response.startswith('```'):

        try:

            response_dict = json.loads(response[3:-3])

            if 'text' in response_dict and 'image' in response_dict:

                return response_dict

        except json.JSONDecodeError:

            pass

    return {"text": "Error", "image": "Error"}

        



def image_agent(query, image_paths, critical_image_info):

    prompt = prompts.image_prompt

    prompt += '\n'

    prompt += f'Question: {query}\n'

    prompt += f'Critical information from another agent: {critical_image_info}\n'

    return invoke_gpt4o(image_paths, prompt)





def text_agent(query, text_data, critical_text_info):

    prompt = prompts.text_prompt

    prompt += '\n'

    prompt += f'Question: {query}\n'

    prompt += f'Critical information from another agent: {critical_text_info}\n'

    prompt += f'Text information: {text_data}\n'

    return invoke_gpt4o(None, prompt)





def sum_agent(query, answer_from_general_agent, answer_from_image_agent, answer_from_text_agent):

    prompt = prompts.sum_prompt

    prompt += '\n'

    prompt += f'Question: {query}\n'

    prompt += f'General answer from general agent: {answer_from_general_agent}\n'

    prompt += f'Image answer from image-only agent: {answer_from_image_agent}\n'

    prompt += f'Text answer from text-only agent: {answer_from_text_agent}\n'

    prompt += 'Your final Answer:'

    response = invoke_gpt4o(None, prompt)

    if response.startswith('{') and response.endswith('}'):

        try:

            response_dict = json.loads(response)

            if 'Answer' in response_dict:

                return response_dict['Answer']

        except json.JSONDecodeError:

            pass

    elif response.startswith('```'):

        try:

            response_dict = json.loads(response[3:-3])

            if 'Answer' in response_dict:

                return response_dict['Answer']

        except json.JSONDecodeError:

            pass

    return "Error"





def process_single_query_item(query_id, m_data: dict):

    """Processes a single query item through the chain of agents."""

    query = m_data["query"]

    retrieved_images_list = m_data["results"].get("retrieved_images")

    if not retrieved_images_list:
        print('Error in retrieved_images_list.')
        exit(1)

    retrieved_texts_list = m_data["results"].get("retrieved_texts")

    if not retrieved_texts_list:
        print('Error in retrieved_texts_list.')
        exit(1)

    retrieved_images_list = retrieved_images_list[:3]
    retrieved_texts_list = retrieved_texts_list[:2]

    retrieved_image_paths = [img_info["image_path"] for img_info in retrieved_images_list if "image_path" in img_info]


    retrieved_text_content = ''

    for t_info in retrieved_texts_list:

        text_path = t_info.get("text_path")

        if not text_path:
            print('Error in text_path.')
            exit(1)

        try:

            with open(text_path, 'r', encoding='utf-8') as f_text:

                retrieved_text_content += f_text.read().strip() + '\n\n' # Add separator

        except FileNotFoundError:

            print(f"Warning: Text file not found at {text_path} for query_id {query_id}")
            exit(1)

        except Exception as e:

            print(f"Warning: Could not read text file {text_path} for query_id {query_id}: {e}")
            exit(1)

    retrieved_text_content = retrieved_text_content.strip()



    current_log_entry = {

        "query_id": query_id,

        "query": query,

        "retrieved_image_paths": retrieved_image_paths,

        "concatenated_retrieved_text_summary": (retrieved_text_content[:500] + "..." if len(retrieved_text_content) > 500 else retrieved_text_content),

    }



    try:

        answer_general = general_agent(query, retrieved_image_paths, retrieved_text_content)

        current_log_entry["answer_from_general_agent"] = answer_general

        

        critical_output = critical_agent(query, retrieved_image_paths, retrieved_text_content, answer_general)

        current_log_entry["critical_agent_output"] = critical_output



        crit_text_info = critical_output.get("text", "Error: 'text' key missing from critical_agent_output")

        crit_image_info = critical_output.get("image", "Error: 'image' key missing from critical_agent_output")



        answer_image = image_agent(query, retrieved_image_paths, crit_image_info)

        current_log_entry["answer_from_image_agent"] = answer_image



        answer_text = text_agent(query, retrieved_text_content, crit_text_info)

        current_log_entry["answer_from_text_agent"] = answer_text



        final_answer = sum_agent(query, answer_general, answer_image, answer_text)

        current_log_entry["final_answer"] = final_answer

    

    except Exception as e:

        print(f"Critical error processing query_id {query_id}: {e}")

        current_log_entry["error_processing_item"] = str(e)

        # Ensure all keys are present for consistent structure

        for key in ["answer_from_general_agent", "critical_agent_output", "answer_from_image_agent", "answer_from_text_agent", "final_answer"]:

            current_log_entry.setdefault(key, "Error: Processing failed due to critical error in item.")

        if "critical_agent_output" not in current_log_entry or not isinstance(current_log_entry["critical_agent_output"], dict):

            current_log_entry["critical_agent_output"] = {"text": "Error", "image": "Error"}





    print(f"Finished processing query_id: {query_id}")

    return current_log_entry



# --- Concurrent Evaluation Function ---

def evaluate_concurrent(query_metadata_iterable, log_filename='answer_log_concurrent.json', max_workers=5):

    """

    Evaluates query metadata concurrently using a ThreadPoolExecutor.

    """
    all_run_logs = []


    # Prepare tasks: (query_id, metadata_item)

    tasks_to_submit = []

    if query_metadata_iterable and isinstance(query_metadata_iterable[0], dict): # List of dicts

        for i, m_data in enumerate(query_metadata_iterable):

            tasks_to_submit.append((m_data.get("query_id_orig", i), m_data)) # Use original ID or index

    else: # Assumed list of (id, dict) tuples

        for query_id, m_data in query_metadata_iterable:

            tasks_to_submit.append((query_id, m_data))

    

    print(f"Submitting {len(tasks_to_submit)} tasks to ThreadPoolExecutor with {max_workers} workers.")



    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_task = {

            executor.submit(process_single_query_item, query_id, m_data): (query_id, m_data)

            for query_id, m_data in tasks_to_submit

        }



        for future in concurrent.futures.as_completed(future_to_task):

            query_id_completed, original_m_data = future_to_task[future]

            try:

                log_entry = future.result()

                all_run_logs.append(log_entry)

            except Exception as exc:

                print(f"Query ID {query_id_completed} generated an unhandled exception in future: {exc}")

                all_run_logs.append({

                    "query_id": query_id_completed,

                    "query": original_m_data.get("query", "N/A"),

                    "error_in_future_result": str(exc)

                })



    # Sort logs by query_id if original order matters for the log file and IDs are sortable

    try:

        all_run_logs.sort(key=lambda x: x.get("query_id", -1))

    except TypeError:

        print("Warning: Could not sort log entries by query_id (possibly mixed types). Logs will be in completion order.")





    try:

        with open(log_filename, 'w', encoding='utf-8') as f:

            json.dump(all_run_logs, f, indent=4)

        print(f"Successfully wrote logs to {log_filename}")

    except IOError as e:

        print(f"Error writing log file {log_filename}: {e}")

    except TypeError as e:

        print(f"Error: Data is not JSON serializable, could not write to {log_filename}: {e}")

        if all_run_logs:

            for i, entry in enumerate(all_run_logs):

                try:

                    json.dumps(entry)

                except TypeError:

                    print(f"Problematic entry at index {i} (query_id {entry.get('query_id', 'UNKNOWN')}):")

                    print(json.dumps(entry, indent=4, default=str)) # Try with default str conversion

                    break

            else:

                 print("Problematic data snippet (first entry, using default str conversion):")

                 print(json.dumps(all_run_logs[0], indent=4, default=str))




# Example usage:

if __name__ == "__main__":

    query_metadata_path = r'evaluation_log_multi.json'

    query_metadata = json.load(open(query_metadata_path, 'r'))

    evaluate_concurrent(query_metadata, log_filename='gen_output.json')
