import os
import json
import re
from tqdm import tqdm
import sys
import time
from llm.llm import LLM
import cv2

dataset = 'ViDoSeek'
img_dir = f'./data/{dataset}/img'
output_dir = f'./data/{dataset}/vlmocr'
os.makedirs(output_dir, exist_ok=True)

prompt = '''Generate bounding boxes for each of the objects in this image in [y_min, x_min, y_max, x_max] format. 
For textual objects, please provide the bounding box for the entire text, and tell me the text content.
For tables, please provide the bounding box for the entire table, and tell me the content in csv format.
For chart, please provide the bounding box for the entire chart, and tell me the content.
For non-textual objects, please provide the bounding box for the object, and tell me the caption of it.


Response Format:
{
    "objects": [
        {
            "bounding_box": [y_min, x_min, y_max, x_max],
            "content": "text content or caption"
            "type": "text" or "table" or "chart" or "object"
        },
        ...
    ],
}
'''

vlm = LLM('your-vlm-name') ## Optional


def draw_boxes(img_path, boxes):
    # boxes: [[y_min, x_min, y_max, x_max], ...]
    img = cv2.imread(img_path)
    # x_len,y len
    x_len,y_len = img.shape[1], img.shape[0]
    for box in boxes:
        cv2.rectangle(img, (int(box[1] * x_len / 1000), int(box[0] * y_len / 1000)), (int(box[3] * x_len / 1000), int(box[2] * y_len / 1000)), (0, 255, 0), 2)
    # save the image
    save_path = img_path.replace('img', 'img_with_boxes_vlmocr')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

def ocr(img_path):
    while True:
        try:
            vlm_output = vlm.generate(query=prompt, image=[img_path])
            vlm_output = vlm_output.replace('```json', '')
            vlm_output = vlm_output.replace('```', '')
            boxes = []
            vlm_output = json.loads(vlm_output)
            for obj in vlm_output['objects']:
                if obj['type'] == 'object':
                    continue
                if 'content' not in obj:
                    raise Exception('not good')
                box = obj['bounding_box']
                boxes.append(box)
            if any([any([x < 0 or x > 1000 for x in box]) for box in boxes]):
                raise Exception('beyond the image')
            draw_boxes(img_path, boxes)
            # save json
            with open(img_path.replace('img', 'vlmocr').replace('.jpg', '.json'), 'w') as f:
                json.dump(vlm_output, f, indent=2, ensure_ascii=False)
            break
        except Exception as e:
            print(f'Error in {img_path}, retrying...')
            time.sleep(2)
        
    


if __name__ == '__main__':
    workers_num = 1
    already_done = [file for file in os.listdir(output_dir) if file.endswith('.json')]
    img_files = [file for file in os.listdir(img_dir) if file.replace('.jpg', '.json') not in already_done]
    if workers_num ==1:
        for img_file in tqdm(img_files):
            img_path = os.path.join(img_dir, img_file)
            ocr(img_path)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers_num) as executor:
            futures = [executor.submit(ocr, os.path.join(img_dir, img_file)) for img_file in img_files]
            for future in tqdm(as_completed(futures), total=len(futures)):
                pass

    
            
