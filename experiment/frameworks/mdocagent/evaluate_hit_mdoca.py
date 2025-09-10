import json


def calculate_hit_rates_single(log_data):
    hit1 = 0
    hit3 = 0
    hit5 = 0
    total = 0
    
    for entry in log_data:
        ground_truth_doc = entry["ground_truth"]["doc_id"]
        ground_truth_pages = set(entry["ground_truth"]["reference_pages"])
        
        retrieved_images = entry["results"]["retrieved_images"]
        retrieved_texts = entry["results"]["retrieved_texts"]
        total += 1
        
        # Hit@1: Check first image only
        if len(retrieved_images) >= 1:
            first_image = retrieved_images[0]
            if (first_image["doc_name"] == ground_truth_doc and 
                first_image["page"] in ground_truth_pages):
                hit1 += 1
        
        # Hit@3: Check first 2 images and first text
        hit3_found = False
        if len(retrieved_images) >= 2 and len(retrieved_texts) >= 1:
            # Check first 2 images
            for img in retrieved_images[:2]:
                if (img["doc_name"] == ground_truth_doc and 
                    img["page"] in ground_truth_pages):
                    hit3_found = True
                    break
            
            # If images didn't hit, check first text
            if not hit3_found:
                first_text = retrieved_texts[0]
                if (first_text["doc_name"] == ground_truth_doc and 
                    first_text["page"] in ground_truth_pages):
                    hit3_found = True
        
        if hit3_found:
            hit3 += 1
        
        # Hit@5: Check first 3 images and first 2 texts
        hit5_found = False
        if len(retrieved_images) >= 3 and len(retrieved_texts) >= 2:
            # Check first 3 images
            for img in retrieved_images[:3]:
                if (img["doc_name"] == ground_truth_doc and 
                    img["page"] in ground_truth_pages):
                    hit5_found = True
                    break
            
            # If images didn't hit, check first 2 texts
            if not hit5_found:
                for text in retrieved_texts[:2]:
                    if (text["doc_name"] == ground_truth_doc and 
                        text["page"] in ground_truth_pages):
                        hit5_found = True
                        break
        
        if hit5_found:
            hit5 += 1
    
    hit1_rate = hit1 / total if total > 0 else 0
    hit3_rate = hit3 / total if total > 0 else 0
    hit5_rate = hit5 / total if total > 0 else 0
    
    return {
        "hit@1": hit1_rate,
        "hit@3": hit3_rate,
        "hit@5": hit5_rate,
        "total_queries": total
    }


def calculate_hit_rates_multi(log_data):
    hit1 = 0
    hit3 = 0
    hit5 = 0
    total = 0
    
    for entry in log_data:
        ground_truth_doc = entry["ground_truth_doc_id"]
        hops = entry["steps"]
        total += 1
        
        retrieved_images = entry["results"]["retrieved_images"]
        retrieved_texts = entry["results"]["retrieved_texts"]
        
        # For each hop, collect the required reference pages
        hop_references = []
        for hop in hops:
            hop_references.append({
                "pages": set(hop["reference_page"]),
                "hit": False
            })
        
        # Hit@1: Check first image only
        if len(retrieved_images) >= 1:
            first_image = retrieved_images[0]
            if first_image["doc_name"] == ground_truth_doc:
                for i, hop_ref in enumerate(hop_references):
                    if first_image["page"] in hop_ref["pages"]:
                        hop_ref["hit"] = True
            
            # Check if all hops hit
            all_hops_hit = all(hop["hit"] for hop in hop_references)
            if all_hops_hit:
                hit1 += 1
        
        # Reset hop hits for next evaluation
        for hop in hop_references:
            hop["hit"] = False
        
        # Hit@3: Check first 2 images and first text
        if len(retrieved_images) >= 2 and len(retrieved_texts) >= 1:
            # Check first 2 images
            for img in retrieved_images[:2]:
                if img["doc_name"] == ground_truth_doc:
                    for i, hop_ref in enumerate(hop_references):
                        if img["page"] in hop_ref["pages"]:
                            hop_ref["hit"] = True
            
            # Check first text
            first_text = retrieved_texts[0]
            if first_text["doc_name"] == ground_truth_doc:
                for i, hop_ref in enumerate(hop_references):
                    if first_text["page"] in hop_ref["pages"]:
                        hop_ref["hit"] = True
            
            # Check if all hops hit
            all_hops_hit = all(hop["hit"] for hop in hop_references)
            if all_hops_hit:
                hit3 += 1
        
        # Reset hop hits for next evaluation
        for hop in hop_references:
            hop["hit"] = False
        
        # Hit@5: Check first 3 images and first 2 texts
        if len(retrieved_images) >= 3 and len(retrieved_texts) >= 2:
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
            if all_hops_hit:
                hit5 += 1
    
    hit1_rate = hit1 / total if total > 0 else 0
    hit3_rate = hit3 / total if total > 0 else 0
    hit5_rate = hit5 / total if total > 0 else 0
    
    return {
        "hit@1": hit1_rate,
        "hit@3": hit3_rate,
        "hit@5": hit5_rate,
        "total_queries": total
    }
if __name__=='__main__':
    log_data_path = r'./evaluation_log_multi_subsampled.json'
    log_data = json.load(open(log_data_path, 'r', encoding='utf-8'))
    hit_rates = calculate_hit_rates_multi(log_data)
    print(hit_rates)
