import json

m3doc_eval_path = r'evaluation_log_multi_aligned.json'
m3doc_eval = json.load(open(m3doc_eval_path, 'r', encoding='utf-8'))

answer_eval_path = r'answer_log_multi_judged.json'

def get_hops_from_query(query):
    for item in m3doc_eval:
        if item["query"] == query:
            return len(item["steps"])
    return 0


two_hops_accuracy = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
three_hops_accuracy = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}

with open(answer_eval_path, 'r', encoding='utf-8') as f:
    data = json.load(f)["results"]
    for entry in data:
        hops = get_hops_from_query(entry["question"])
        if 'error' in entry["evaluation"]:
            continue
        try:
            if hops == 2:
                two_hops_accuracy["total"] += 1
                if entry["evaluation"]["score"] >= 7:
                    two_hops_accuracy["correct"] += 1
                elif entry["evaluation"]["score"] >= 4:
                    two_hops_accuracy["partial"] += 1
                else:
                    two_hops_accuracy["incorrect"] += 1
            elif hops == 3:
                three_hops_accuracy["total"] += 1
                if entry["evaluation"]["score"] >= 7:
                    three_hops_accuracy["correct"] += 1
                elif entry["evaluation"]["score"] >= 4:
                    three_hops_accuracy["partial"] += 1
                else:
                    three_hops_accuracy["incorrect"] += 1
        except:
            print(entry)
            exit(1)

print(f'Two-hop query accuracy: {two_hops_accuracy["correct"] / two_hops_accuracy["total"] if two_hops_accuracy["total"] > 0 else 0:.3f} ({two_hops_accuracy["correct"]}/{two_hops_accuracy["total"]})')
print(f'Two-hop query partial accuracy: {two_hops_accuracy["partial"] / two_hops_accuracy["total"] if two_hops_accuracy["total"] > 0 else 0:.3f} ({two_hops_accuracy["partial"]}/{two_hops_accuracy["total"]})')
print(f'Two-hop query incorrect accuracy: {two_hops_accuracy["incorrect"] / two_hops_accuracy["total"] if two_hops_accuracy["total"] > 0 else 0:.3f} ({two_hops_accuracy["incorrect"]}/{two_hops_accuracy["total"]})')

print(f'Three-hop query accuracy: {three_hops_accuracy["correct"] / three_hops_accuracy["total"] if three_hops_accuracy["total"] > 0 else 0:.3f} ({three_hops_accuracy["correct"]}/{three_hops_accuracy["total"]})')
print(f'Three-hop query partial accuracy: {three_hops_accuracy["partial"] / three_hops_accuracy["total"] if three_hops_accuracy["total"] > 0 else 0:.3f} ({three_hops_accuracy["partial"]}/{three_hops_accuracy["total"]})')
print(f'Three-hop query incorrect accuracy: {three_hops_accuracy["incorrect"] / three_hops_accuracy["total"] if three_hops_accuracy["total"] > 0 else 0:.3f} ({three_hops_accuracy["incorrect"]}/{three_hops_accuracy["total"]})')
