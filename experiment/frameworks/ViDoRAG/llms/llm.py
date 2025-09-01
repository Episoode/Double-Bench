import torch
from PIL import Image
from pathlib import Path
import sys
import base64
from io import BytesIO
import os


def _encode_image(image_path):
    if isinstance(image_path,Image.Image):
        buffered = BytesIO()
        image_path.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
        base64_encoded = base64.b64encode(img_data).decode("utf-8")
        return base64_encoded
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

class Qwen_VL_2_5:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto",attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def generate(self,query, images):
        from qwen_vl_utils import process_vision_info
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        content = [dict(
            type = "image",
            image = img
        ) for img in images]
        content.append(dict(type='text',text=query))
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1028)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
    
        
class LLM:
    def __init__(self,model_name):
        self.model_name = model_name
        if 'Qwen2.5-VL' in self.model_name:
            self.model = Qwen_VL_2_5(model_name)
        elif model_name.startswith('gpt'):
            from openai import OpenAI
            os.environ['OPENAI_API_KEY'] = 'your-api-key'
            self.model = OpenAI()
            
    def generate(self,**kwargs):
        query = kwargs.get('query','')
        image = kwargs.get('image','')
        model_name = kwargs.get('model_name','')

        if 'Qwen2.5-VL' in self.model_name:
            return self.model.generate(query,image)
        elif self.model_name.startswith('gpt'):
            content = [{
                "type": "text",
                "text": query
            }]
            if image != '':
                filepaths = [Path(img).resolve().as_posix() for img in image]
                for filepath in filepaths:
                    base64_image = _encode_image(filepath)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}}
                        )
            completion = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                    "role": "user",
                    "content": content
                    }
                ])
            return completion.choices[0].message.content

if __name__ == '__main__':
    llm = LLM('gpt-4o')
    response = llm.generate(query='describe in 3 words',image=['image_path'])
    print(response)