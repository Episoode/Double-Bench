import os
import json
# read
import argparse
import math
from tqdm import tqdm
def recall_at_k(relevant_docs, retrieved_docs, k):
    recall_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            ret_set = set(ret)
        else:
            ret_set = set(ret[:k])
        rel_set = set(rel)
        recall = len(rel_set & ret_set) / len(rel_set)
        recall_values.append(recall)
    return sum(recall_values) / len(recall_values)

def hit_rate_at_k(relevant_docs, retrieved_docs, k):
    hit_count = 0
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            k = len(ret)
        if any(doc in set(ret[:k]) for doc in rel):
            hit_count += 1
    return hit_count / len(relevant_docs)

def ndcg_at_k(relevant_docs, retrieved_docs, k):
    def dcg(scores):
        return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))

    ndcg_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            k = len(ret)
        rel_set = set(rel)
        #  DCG@k
        scores = [1 if doc in rel_set else 0 for doc in ret[:k]]
        dcg_value = dcg(scores)
        #  IDCG@k
        ideal_scores = sorted([1] * len(rel_set) + [0] * (k - len(rel_set)), reverse=True)[:k]
        idcg_value = dcg(ideal_scores)
        ndcg = dcg_value / idcg_value if idcg_value > 0 else 0
        ndcg_values.append(ndcg)
    return sum(ndcg_values) / len(ndcg_values)

def mrr_at_k(relevant_docs, retrieved_docs, k):
    rr_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            k = len(ret)
        for rank, doc in enumerate(ret[:k], start=1):
            if doc in rel:
                rr_values.append(1 / rank)
                break
        else:
            rr_values.append(0)
    return sum(rr_values) / len(rr_values)

def eval_sample(example):
    retrieved_docs_list = []
    relevant_docs_list = []
    
    file_name = example['meta_info']['file_name']

def eval_search(examples):
    relevant_docs_list = []
    retrieved_docs_list = []
    results = {}
    if examples[0].get('recall_results',None) is not None:
        for example in tqdm(examples):
            file_name = example['meta_info']['file_name']
            file_name = file_name.split('.')[0]
            reference_page = example['meta_info']['reference_page']
            reference_page = [str(page) for page in reference_page]
            relevant_docs_list.append([file_name + '_' + page for page in reference_page])
            retrieved_docs = []
            if 'source_nodes' in example['recall_results']:
                for node in example['recall_results']['source_nodes']:
                    if 'node' in node:
                        retrieved_docs.append(os.path.basename(node['node']['metadata'].get('filename',node['node']['metadata'].get('file_name'))))
                    else:
                        retrieved_docs.append(os.path.basename(node['metadata']['file_path']))
            else:
                for node in example['recall_results']:
                    if 'node' in node:
                        retrieved_docs.append(os.path.basename(node['node']['metadata'].get('filename',node['node']['metadata'].get('file_name'))))
                    else:
                        retrieved_docs.append(os.path.basename(node['metadata']['file_path']))
            # retrieved_docs = [os.path.basename(node['node']['metadata']['filename']) for node in example['recall_results']['source_nodes']]
            retrieved_docs_list.append([doc.split('.')[0] for doc in retrieved_docs])

        for k in [-1,1,5,10]:
            recall = recall_at_k(relevant_docs_list, retrieved_docs_list, k)
            hit_rate = hit_rate_at_k(relevant_docs_list, retrieved_docs_list, k)
            ndcg = ndcg_at_k(relevant_docs_list, retrieved_docs_list, k)
            mrr = mrr_at_k(relevant_docs_list, retrieved_docs_list, k)
            if k == -1:
                k = 'all'
            results[f'Recall@{k}'] = recall
            results[f'Hit rate@{k}'] = hit_rate
            results[f'nDCG@{k}'] = ndcg
            results[f'MRR@{k}'] = mrr

    if examples[0].get('eval_result',None) is not None:
        score = 0
        passing = 0
        for example in examples:
            score += example['eval_result']['score']
            passing += example['eval_result']['passing']
        results.update(dict(
            score=score / len(examples),
            passing=passing / len(examples)
        ))
    return results

def eval_search_type_wise(examples):
    examples_by_type = {}
    for example in examples:
        query_type = example['meta_info'].get('query_type', '')
        source_type = example['meta_info'].get('source_type', '')
        if '-Hop' in query_type:
            source_type = query_type.split('_')[-1]
            query_type = query_type.split('_')[0]
        if query_type != '':
            if query_type not in examples_by_type:
                examples_by_type[query_type] = []
            examples_by_type[query_type].append(example)
        if source_type != '':
            if source_type not in examples_by_type:
                examples_by_type[source_type] = []
            examples_by_type[source_type].append(example)
    results = {}
    for query_type, examples in examples_by_type.items():
        results[query_type] = eval_search(examples)
    return results
