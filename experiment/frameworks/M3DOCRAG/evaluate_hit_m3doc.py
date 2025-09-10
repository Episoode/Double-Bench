import json

def calculate_hit_rates_single(log_data):
    """
    Calculate hit@1, hit@3, and hit@5 rates from the log data.
    
    Args:
        log_data: List of dictionaries containing the log entries
        
    Returns:
        Dictionary containing hit rates for 1, 3, and 5
    """
    hit1_count = 0
    hit3_count = 0
    hit5_count = 0
    total_entries = len(log_data)
    
    for entry in log_data:
        # Get ground truth pages with doc_ids
        gt_pages = set((entry['ground_truth_doc_id'], page) 
                       for page in entry['ground_truth_pages'])
        
        # Check hit@1
        top1 = entry['retrieved_top_k_pages'][:1]
        for item in top1:
            if (item['doc_id'], item['page']) in gt_pages:
                hit1_count += 1
                break
                
        # Check hit@3
        top3 = entry['retrieved_top_k_pages'][:3]
        hit3 = False
        for item in top3:
            if (item['doc_id'], item['page']) in gt_pages:
                hit3 = True
                break
        if hit3:
            hit3_count += 1
            
        # Check hit@5
        top5 = entry['retrieved_top_k_pages'][:5]
        hit5 = False
        for item in top5:
            if (item['doc_id'], item['page']) in gt_pages:
                hit5 = True
                break
        if hit5:
            hit5_count += 1
    
    return {
        'hit@1': hit1_count / total_entries,
        'hit@3': hit3_count / total_entries,
        'hit@5': hit5_count / total_entries,
        'total_entries': total_entries
    }


def calculate_hit_rates_multi(log_data):
    """
    Calculate multi-hop hit@1, hit@3, and hit@5 rates.
    A hit is counted only if ALL hops have at least one correct retrieved page.
    
    Args:
        log_data: List of dictionaries containing multi-hop log entries
        
    Returns:
        Dictionary containing hit rates for 1, 3, and 5
    """
    hit1_count = 0
    hit3_count = 0
    hit5_count = 0
    total_entries = len(log_data)
    
    for entry in log_data:
        # Get ground truth pages for each hop (list of sets)
        hop_evidence_sets = []
        for hop in entry['hop_steps_info']:
            hop_pages = set((entry['ground_truth_doc_id'], page) 
                         for page in hop['reference_page'])
            hop_evidence_sets.append(hop_pages)
        
        # Check hit@1
        top1 = entry['retrieved_top_k_pages'][:1]
        hit1 = all(any((item['doc_id'], item['page']) in hop_set 
                      for item in top1) 
                  for hop_set in hop_evidence_sets)
        if hit1:
            hit1_count += 1
            
        # Check hit@3
        top3 = entry['retrieved_top_k_pages'][:3]
        hit3 = all(any((item['doc_id'], item['page']) in hop_set 
                  for item in top3) 
                  for hop_set in hop_evidence_sets)
        if hit3:
            hit3_count += 1
            
        # Check hit@5
        top5 = entry['retrieved_top_k_pages'][:5]
        hit5 = all(any((item['doc_id'], item['page']) in hop_set 
                  for item in top5) 
                  for hop_set in hop_evidence_sets)
        if hit5:
            hit5_count += 1
    
    return {
        'hit@1': hit1_count / total_entries if total_entries > 0 else 0,
        'hit@3': hit3_count / total_entries if total_entries > 0 else 0,
        'hit@5': hit5_count / total_entries if total_entries > 0 else 0,
        'total_entries': total_entries
    }


if __name__=='__main__':
    log_data_path = r'./m3doc_rag_storage/rag_multihop_eval_details_20250724_041850.json'
    log_data = json.load(open(log_data_path, 'r', encoding='utf-8'))
    hit_rates = calculate_hit_rates_multi(log_data)
    print(hit_rates)
