import json
from agents.mdoc_agent import MDocAgent

class MDAi(MDocAgent):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, sample, question, texts, images):
        general_agent = self.agents[-1]
        general_response, messages = general_agent.predict(question, texts, images, with_sys_prompt=True)
        critical_info = general_agent.self_reflect(prompt = general_agent.config.agent.critical_prompt, add_to_message=False)

        start_index = critical_info.find('{') 
        end_index = critical_info.find('}') + 1 
        critical_info = critical_info[start_index:end_index]
        text_info = ""
        image_info = ""
        try:
            critical_info = json.loads(critical_info)
            text_info = critical_info.get("text", "")
            image_info = critical_info.get("image", "")
        except Exception as e:
            print(e)

        image_agent = self.agents[0]
        all_messages = "General Agent:\n" + general_response + "\n"
        
        relect_prompt = "\nYou may use the given clue:\n"

        image_response, messages = image_agent.predict(question + relect_prompt + image_info, texts = None, images = images, with_sys_prompt=True)
        all_messages += "Image Agent:\n" + image_response + "\n"
            
        final_ans, final_messages = self.sum(all_messages)
        
        return final_ans, final_messages
    
class MDAt(MDocAgent):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, sample, question, texts, images):
        general_agent = self.agents[-1]
        outputs, messages = general_agent.predict(question, texts, images, with_sys_prompt=True)
        critical_info = general_agent.self_reflect(prompt = general_agent.config.agent.critical_prompt, add_to_message=False)

        start_index = critical_info.find('{') 
        end_index = critical_info.find('}') + 1 
        critical_info = critical_info[start_index:end_index]
        text_info = ""
        image_info = ""
        try:
            critical_info = json.loads(critical_info)
            text_info = critical_info.get("text", "")
            image_info = critical_info.get("image", "")
        except Exception as e:
            print(e)

        text_agent = self.agents[1]
        all_messages = "General Agent:\n" + outputs + "\n"
        
        relect_prompt = "\nYou may use the given clue:\n"
        text_response, messages = text_agent.predict(question + relect_prompt + text_info, texts = texts, images = None, with_sys_prompt=True)
        all_messages += "Text Agent:\n" + text_response + "\n"

        final_ans, final_messages = self.sum(all_messages)
        
        return final_ans, final_messages
    
class MDAs(MDocAgent):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, sample, question, texts, images):
        text_agent = self.agents[1]
        image_agent = self.agents[0]
        all_messages = ""
        
        text_response, messages = text_agent.predict(question, texts = texts, images = None, with_sys_prompt=True)
        all_messages += "Text Agent:\n" + text_response + "\n"
        image_response, messages = image_agent.predict(question, texts = None, images = images, with_sys_prompt=True)
        all_messages += "Image Agent:\n" + image_response + "\n"
            
        final_ans, final_messages = self.sum(all_messages)
        
        return final_ans, final_messages