from PIL import Image
import os
import json
import sys

from agent.agent_prompt import seeker_prompt,inspector_prompt,answer_prompt
from agent.map_dict import arrangement_map_dict,page_map_dict_normal,page_map_dict

from utils.parse_tool import extract_json
from utils.image_preprosser import concat_images_with_bbox

class Seeker:
    def __init__(self, vlm):
        self.vlm = vlm
        self.seeker_multi_image = True

        if self.seeker_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict
    def run(self, query=None, images_path=None, feedback=None):

        if query is not None and images_path is not None:
            self.buffer_images = images_path
            self.query = query
            prompt = seeker_prompt.replace('{question}', self.query).replace('{page_map}', self.page_map[len(self.buffer_images)])
        
        elif feedback is not None:
            additional_information = self.query + '\n\n## Additional Information\n' + feedback
            prompt = seeker_prompt.replace('{question}', additional_information).replace('{page_map}', self.page_map[len(self.buffer_images)])        

        input_images = self.buffer_images

        times = 0
        while True:
            if times > 2:
                # return None, None, None
                raise Exception('seeker time out')
            times += 1
            select_response = self.vlm.generate(query=prompt, image=input_images)
            # print(select_response)
            try:
                select_response_json = extract_json(select_response)
                reason = select_response_json.get('reason', None)
                summary = select_response_json.get('summary', None)
                select_page_num = select_response_json.get('choice', None)
                if reason is None or summary is None or any([page >= len(self.buffer_images) for page in select_page_num]):
                    raise Exception(f'select json format error: length: {len(self.buffer_images)}')

                selected_images = [self.buffer_images[page] for page in select_page_num]

                self.buffer_images = [img for i, img in enumerate(self.buffer_images) if i not in select_page_num]

            except Exception as e:
                # print(e)
                # print(select_response)
                # print('seeker')
                continue
            break
        
        return selected_images, summary, reason

class Inspector:
    def __init__(self, vlm):
        self.vlm = vlm
        self.inspector_multi_image = True

        if self.inspector_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

        self.buffer_images = []

    def run(self, query, images_path):
        # answer or not, (images, candidate)/feedback
        if len(self.buffer_images) == 0 and len(images_path) == 0:
            return None, None, None
        elif len(images_path) == 0:
            return 'synthesizer', None, self.buffer_images
        elif len(images_path) != 0:
            self.buffer_images.extend(images_path)

        input_images = self.buffer_images

        prompt = inspector_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(self.buffer_images)])

        times = 0
        while True:
            if times >2:
                raise Exception('Inspector time out')
                # return None, None, None
            times +=1

            response = self.vlm.generate(query=prompt,image=input_images)
            # print(response)
            try:
                response_json = extract_json(response)
                # thought
                reason = response_json.get('reason',None)
                # if feedback
                info = response_json.get('information',None)
                choice = response_json.get('choice',None)
                # can answer
                answer = response_json.get('answer',None)
                ref = response_json.get('reference',None)

                if reason is None:
                    raise Exception('answer no reason')
                elif answer is not None and ref is not None:
                    if any([page >= len(self.buffer_images) for page in ref]) or len(ref)==0:
                        raise Exception('ref error')
                    if len(ref) == len(self.buffer_images):
                        return 'answer', answer, self.buffer_images
                    else:
                        ref_images = [self.buffer_images[page] for page in ref]
                        return 'synthesizer', answer, ref_images
                elif info is not None and choice is not None:
                    if any([page >= len(self.buffer_images) for page in choice]):
                        raise Exception('choice error')

                    self.buffer_images = [self.buffer_images[page] for page in choice]
                    return 'seeker', info, self.buffer_images
            
            except Exception as e:
                # print(e)
                # print("inspector")
                continue

class Synthesizer:
    def __init__(self, vlm):
        self.vlm = vlm
        self.synthesizer_multi_image = True
        if self.synthesizer_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

    def run(self, query, candidate_answer, ref_images):
        if candidate_answer is not None:
            query = query + '\n\n## Related Information\n' + candidate_answer
        prompt = answer_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(ref_images)])

        input_images = ref_images
        
        while True:
            final_answer_response = self.vlm.generate(query=prompt,image=input_images)
            # print(final_answer_response)
            try:
                final_answer_response_json = extract_json(final_answer_response)
                reason = final_answer_response_json.get('reason',None)
                answer = final_answer_response_json.get('answer',None)
                if reason is None or answer is None :
                    raise Exception('Synthesizer time out')
                return reason, answer
            except Exception as e:
                # print(e)
                # print(final_answer_response)
                # print("answer")
                continue

class ViDoRAG_Agents:
    def __init__(self, vlm):
        self.seeker = Seeker(vlm)
        self.inspector = Inspector(vlm)
        self.synthesizer = Synthesizer(vlm)

    def run_agent(self, query, images_path):
        # initial
        self.seeker.buffer_images = []
        self.inspector.buffer_images = []

        selected_images, summary, reason = self.seeker.run(query=query, images_path=images_path)
        # iter
        while True:
            status, information, images = self.inspector.run(query, selected_images)
            if status == 'answer':
                return information
            elif status == 'synthesizer':
                reason, answer = self.synthesizer.run(query, information, images)
                return answer
            elif status == 'seeker':
                selected_images, summary, reason = self.seeker.run(feedback=information)
                continue
            else:
                print('No related information')
                try:
                    fallback_answer = self.synthesizer.vlm.generate(query=query, image=images_path)
                except Exception as e:
                    fallback_answer = f"Last API Failed: {str(e)}"
                return fallback_answer

#test example
if __name__ == '__main__':
    from llms.llm import LLM
    vlm = LLM('gpt-4o')
    agent = ViDoRAG_Agents(vlm)
    re=agent.run_agent(query='Who is Tim?', images_path=['/path/to/your/eg1.jpg','/path/to/your/eg2.jpg','/path/to/your/eg3.jpg','/path/to/your/eg4.jpg'])

