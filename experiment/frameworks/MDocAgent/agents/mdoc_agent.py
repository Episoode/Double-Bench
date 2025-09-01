
from tqdm import tqdm
import importlib
import json
import torch
import os
from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset

class MDocAgent(MultiAgentSystem):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, question, texts, images):
        general_agent = self.agents[-1]
        general_response, messages = general_agent.predict(question, texts, images, with_sys_prompt=True)
        # print("### General Agent: "+ general_response)
        critical_info = general_agent.self_reflect(prompt = general_agent.config.agent.critical_prompt, add_to_message=False)
        # print("### General Critical Agent: " + critical_info)

        start_index = critical_info.find('{') 
        end_index = critical_info.find('}') + 1 
        critical_info = critical_info[start_index:end_index]
        text_reflection = ""
        image_reflection = ""
        try:
            critical_info = json.loads(critical_info)
            text_reflection = critical_info.get("text", "")
            image_reflection = critical_info.get("image", "")
        except Exception as e:
            print(e)

        text_agent = self.agents[1]
        image_agent = self.agents[0]
        all_messages = "General Agent:\n" + general_response + "\n"
        
        relect_prompt = "\nYou may use the given clue:\n"
        text_response, messages = text_agent.predict(question + relect_prompt +text_reflection, texts = texts, images = None, with_sys_prompt=True)
        all_messages += "Text Agent:\n" + text_response + "\n"
        image_response, messages = image_agent.predict(question + relect_prompt +image_reflection, texts = None, images = images, with_sys_prompt=True)
        all_messages += "Image Agent:\n" + image_response + "\n"
            
        # print("### Text Agent: " + text_response)
        # print("### Image Agent: " + image_response)
        final_ans, final_messages = self.sum(all_messages)
        # print("### Final Answer: "+final_ans)
        
        return final_ans, final_messages
