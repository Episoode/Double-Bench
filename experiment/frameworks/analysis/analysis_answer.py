import json
from openai import OpenAI
from tqdm import tqdm
import asyncio

client = OpenAI(
    api_key="fake-key",
)

def test():
    prompt = f"""
    Analyze the following response and determine if it constitutes a refusal to answer the question.
    A refusal typically includes phrases indicating inability to answer, lack of information, explicit statements of not knowing, or suggesting the user to consult professionals or visit official web pages for more detailed answers.

    Here are some examples of refusal:
    1. La empresa HNZ no ha proporcionado un n\u00famero espec\u00edfico de alumnos formados en DevOps. Se destaca su participaci\u00f3n en comunidades y eventos relacionados, pero para obtener informaci\u00f3n precisa, se recomienda consultar directamente con HNZ o su sitio web.
    2. The provided pages do not contain relavent information to answer the query.
    3. The document did not provide specific answers for the query.
    
    Response: "I cannot find any information from given pages."
    
    Your analysis should be strictly 'yes' if it is a refusal explained by the above rules, or 'no' otherwise.
    """

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt + ' /no_think'}]}]

    response = client.chat.completions.create(
            model="Qwen3-32B",
            messages=messages,
            temperature=0.0,
            max_tokens=10
        )
    judgment = response.choices[0].message.content.lower().strip()
    print(judgment)

def is_refusal_llm(answer):
    """Use LLM to determine if the answer is a refusal to answer the query"""

    if isinstance(answer, list):
        answer = '\n'.join(str(answer))
    elif not isinstance(answer, str):
        try:
            answer = str(answer)
        except:
            print(answer)
            exit(1)
    
    if not answer or not answer.strip():
        return True
    
    prompt = f"""
    Analyze the following response and determine if it constitutes a refusal to answer the question.
    A refusal typically includes phrases indicating inability to answer, lack of information, explicit statements of not knowing, or suggesting the user to consult professionals or visit official web pages for more detailed answers.

    Here are some examples of refusal:
    1. La empresa HNZ no ha proporcionado un n\u00famero espec\u00edfico de alumnos formados en DevOps. Se destaca su participaci\u00f3n en comunidades y eventos relacionados, pero para obtener informaci\u00f3n precisa, se recomienda consultar directamente con HNZ o su sitio web.
    2. The provided pages do not contain relavent information to answer the query.
    3. The document did not provide specific answers for the query.
    
    Response: "{answer}"
    
    Your analysis should be strictly 'yes' if it is a refusal explained by the above rules, or 'no' otherwise.
    """
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt + ' /no_think'}]}]
    
    try:
        response = client.chat.completions.create(
            model="Qwen3-32B",
            messages=messages,
            temperature=0.0,
            max_tokens=10
        )
        judgment = response.choices[0].message.content.lower().strip()
        return 'yes' in judgment or 'Yes' in judgment
    except Exception as e:
        print(f"Error judging refusal: {e}")
        # Fallback to simple rule if LLM fails
        return any(phrase.lower() in answer.lower() for phrase in [
            "i don't know", "i don't have", "cannot answer", "unable to provide",
            "no information", "not mentioned", "not provided", "not found"
        ])

def if_hit_single(entry):
    ground_truth_doc = entry["ground_truth"]["doc_id"] if "ground_truth" in entry else entry["ground_truth_doc_id"]
    ground_truth_pages = set(entry["ground_truth"]["reference_pages"]) if "ground_truth" in entry else set(entry["ground_truth_pages"])
        
    retrieved_images = entry["results"]["retrieved_images"] if "results" in entry else entry["retrieved_top_k_pages"]
    retrieved_texts = entry["results"]["retrieved_texts"] if "results" in entry else None
        
    # Hit@5: Check first 3 images and first 2 texts
    if retrieved_texts is None:
        for img in retrieved_images[:5]:
            if (img["doc_id"] == ground_truth_doc and 
                img["page"] in ground_truth_pages):
                return True
    elif len(retrieved_images) >= 3 and len(retrieved_texts) >= 2:
        # Check first 3 images
        for img in retrieved_images[:3]:
            if (img["doc_name"] == ground_truth_doc and 
                img["page"] in ground_truth_pages):
                return True
            
        for text in retrieved_texts[:2]:
            if (text["doc_name"] == ground_truth_doc and 
                text["page"] in ground_truth_pages):
                return True
    
    return False


def if_hit_multi(entry):
    ground_truth_doc = entry["ground_truth_doc_id"]
    hops = entry["steps"] if "steps" in entry else entry["hop_steps_info"]
        
    retrieved_images = entry["results"]["retrieved_images"] if "results" in entry else entry["retrieved_top_k_pages"]
    retrieved_texts = entry["results"]["retrieved_texts"] if "results" in entry else None
        
    # For each hop, collect the required reference pages
    hop_references = []
    for hop in hops:
        hop_references.append({
                "pages": set(hop["reference_page"]),
                "hit": False
        })
        
    # Hit@5: Check first 3 images and first 2 texts
    if retrieved_texts is None:
        # Check first 5 images
        for img in retrieved_images[:5]:
            if img["doc_id"] == ground_truth_doc:
                for i, hop_ref in enumerate(hop_references):
                    if img["page"] in hop_ref["pages"]:
                        hop_ref["hit"] = True
        # Check if all hops hit
        all_hops_hit = all(hop["hit"] for hop in hop_references)
    elif len(retrieved_images) >= 3 and len(retrieved_texts) >= 2:
        # Check first 3 images
        for img in retrieved_images[:3]:
            if img["doc_name"] == ground_truth_doc:
                for i, hop_ref in enumerate(hop_references):
                    if img["page"] in hop_ref["pages"]:
                        hop_ref["hit"] = True
            
        # Check first 2 texts
        for text in retrieved_texts[:2]:
            if text["doc_name"] == ground_truth_doc:
                for i, hop_ref in enumerate(hop_references):
                    if text["page"] in hop_ref["pages"]:
                        hop_ref["hit"] = True
            
        # Check if all hops hit
        all_hops_hit = all(hop["hit"] for hop in hop_references)
    return all_hops_hit

async def process_batch(batch_r, batch_a, sin_or_mul):
    """Process a batch of entries synchronously"""
    batch_results = []
    for r_entry, a_entry in zip(batch_r, batch_a):
        if sin_or_mul:
            hit = if_hit_single(r_entry)
        else:
            hit = if_hit_multi(r_entry)
        answer = a_entry["final_answer"] if "final_answer" in a_entry else a_entry["generated_answer"]
        refusal = is_refusal_llm(answer)
        
        # Determine the label
        if not hit and refusal:
            label = "no_hit_no_answer"
        elif not hit and not refusal:
            label = "no_hit_answer"
        elif hit and refusal:
            label = "hit_no_answer"
        else:
            label = "hit_answer"

        if "ground_truth" in r_entry:
            ground_truth = r_entry["ground_truth"]
        elif "steps" in r_entry:
            ground_truth = r_entry["steps"]
        elif "ground_truth_pages" in a_entry:
            ground_truth = a_entry["ground_truth_pages"]
        elif "hop_steps_info" in a_entry:
            ground_truth = a_entry["hop_steps_info"]
        else:
            print("ground_truth entry not found. Exiting.")
            exit(1)
            
        batch_results.append({
            "query": a_entry.get("query", "") if "query" in a_entry else a_entry.get("question", ""),
            "answer": answer,
            "hit": hit,
            "refusal": refusal,
            "label": label,
            "retrieved_images": r_entry["results"]["retrieved_images"][:3] if "results" in r_entry else r_entry["retrieved_top_k_pages"][:5],
            "retrieved_texts": r_entry["results"]["retrieved_texts"][:2] if "results" in r_entry else None,
            "ground_truth": ground_truth,
            "reference_answer": a_entry["reference_answer"] if "reference_answer" in a_entry else a_entry["ground_truth_answer_reference"]
        })
    return batch_results

async def analysis(retrieve_log_file, answer_log_file, sin_or_mul, output_path="llm_analysis_results.json"):
    '''
    Analyze the log files using LLM for refusal detection with async batching
    '''
    no_hit_no_answer = 0
    no_hit_answer = 0
    hit_no_answer = 0
    hit_answer = 0
    
    results = []
    
    # Process entries in batches to manage load
    batch_size = 8
    for i in tqdm(range(0, len(retrieve_log_file), batch_size)):
        batch_r = retrieve_log_file[i:i+batch_size]
        batch_a = answer_log_file[i:i+batch_size]
        
        # Process each batch synchronously within the async loop
        batch_results = await process_batch(batch_r, batch_a, sin_or_mul)
        
        for result in batch_results:
            label = result["label"]
            if label == "no_hit_no_answer":
                no_hit_no_answer += 1
            elif label == "no_hit_answer":
                no_hit_answer += 1
            elif label == "hit_no_answer":
                hit_no_answer += 1
            else:
                hit_answer += 1
            results.append(result)
    
    # Print summary statistics
    print("\nAnalysis Results:")
    print(f"no_hit_no_answer: {no_hit_no_answer}")
    print(f"no_hit_answer: {no_hit_answer}")
    print(f"hit_no_answer: {hit_no_answer}")
    print(f"hit_answer: {hit_answer}")
    
    total = len(retrieve_log_file)
    print("\nPercentages:")
    print(f"no_hit_no_answer: {no_hit_no_answer/total:.2%}")
    print(f"no_hit_answer: {no_hit_answer/total:.2%}")
    print(f"hit_no_answer: {hit_no_answer/total:.2%}")
    print(f"hit_answer: {hit_answer/total:.2%}")
    
    # Save results to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "no_hit_no_answer": no_hit_no_answer,
                "no_hit_answer": no_hit_answer,
                "hit_no_answer": hit_no_answer,
                "hit_answer": hit_answer,
                "total": total
            },
            "detailed_results": results
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

async def main():
    retrieve_log_path = ''
    answer_log_path = ''
    
    print("Loading log files...")
    with open(retrieve_log_path, 'r', encoding='utf-8') as f:
        retrieve_log_file = json.load(f)
        sin_or_mul = not ('multi' in retrieve_log_path)
    with open(answer_log_path, 'r', encoding='utf-8') as f:
        answer_log_file = json.load(f)
    
    print(f"Loaded {len(retrieve_log_file)} entries for analysis")
    await analysis(retrieve_log_file, answer_log_file, sin_or_mul, output_path = "llm_analysis_m3doc_m.json")

if __name__ == '__main__':
    asyncio.run(main())
