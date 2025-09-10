## Examples
```
#Visual & Multimodal Embedding Models
python colqwen.py --mode embed --root_dir docs --output_dir colqwen_store
python colqwen.py --mode search --output_dir colqwen_store --query "your query"
python colqwen.py --mode process_json --output_dir colqwen_store --json_file single_bench.json --output_json colqwen_single.json --top_k 10

#Text Embedding Models
python nv.py --mode embed --ocr_dir /path/to/ocr --output_dir /path/to/nv_store
python nv.py --mode search --output_dir /path/to/nv_store --query "your query"
python nv.py --mode process_json --output_dir /path/to/nv_store --json_file single_bench.json --output_json nv_single.json
```



