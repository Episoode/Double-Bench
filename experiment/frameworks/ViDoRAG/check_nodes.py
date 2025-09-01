import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict


def load_node_file(file_path):
    """Load the content of a single .node file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def count_files_by_id(directory):
    """Count the number of files associated with each id_"""
    node_files = list(Path(directory).glob('**/*.node'))
    print(f"Found {len(node_files)} node files")

    id_counts = defaultdict(list)
    for file in tqdm(node_files, desc="Counting files by id"):
        content = load_node_file(file)
        if content and 'metadata' in content and 'id_' in content['metadata']:
            id_value = content['metadata']['id_']
            id_counts[id_value].append(file)

    sorted_counts = sorted(id_counts.items(), key=lambda x: len(x[1]), reverse=True)

    print("\n=== ID Statistics ===")
    print(f"Total unique IDs: {len(id_counts)}")
    print("\nTop 10 IDs with most files:")
    for id_value, files in sorted_counts[:10]:
        print(f"\nID: {id_value}")
        print(f"Number of files: {len(files)}")
        print("Files:")
        for file in files:
            print(f"  {file}")

    count_distribution = defaultdict(int)
    for id_value, files in id_counts.items():
        count_distribution[len(files)] += 1

    print("\n=== File Count Distribution ===")
    print("Number of files per ID | Number of IDs")
    print("----------------------------------------")
    for count in sorted(count_distribution.keys()):
        print(f"{count:^20} | {count_distribution[count]:^12}")


def compare_nodes(node1, node2):
    """Compare the content of two node files"""
    if node1 is None or node2 is None:
        return False
    keys_to_compare = ['text', 'metadata', 'embedding']
    for key in keys_to_compare:
        if key in node1 and key in node2:
            if node1[key] != node2[key]:
                return False
    return True


def find_different_nodes(directory):
    """Find different node files in a directory"""
    node_files = list(Path(directory).glob('**/*.node'))
    print(f"Found {len(node_files)} node files")

    file_groups = defaultdict(list)
    for file in node_files:
        file_groups[file.name].append(file)

    different_files = []
    for filename, files in tqdm(file_groups.items(), desc="Comparing files"):
        if len(files) > 1:
            reference = load_node_file(files[0])
            if reference is None:
                continue
            for other_file in files[1:]:
                other_content = load_node_file(other_file)
                if not compare_nodes(reference, other_content):
                    different_files.append((files[0], other_file))
    return different_files


def main():
    parser = argparse.ArgumentParser(description='Check differences between node files')
    parser.add_argument('directory', help='Directory containing node files, e.g. path/to/your/node_dir')
    parser.add_argument('--mode', choices=['compare', 'count'], default='count',
                        help='Mode: compare files or count by id (default: count)')
    args = parser.parse_args()

    print(f"Checking node files in: {args.directory}")

    if args.mode == 'count':
        count_files_by_id(args.directory)
    else:
        different_files = find_different_nodes(args.directory)
        if different_files:
            print("\nFound different files:")
            for file1, file2 in different_files:
                print(f"\nDifferent pair:")
                print(f"  {file1}")
                print(f"  {file2}")
        else:
            print("\nNo different files found!")


if __name__ == '__main__':
    main()