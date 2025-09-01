import json
import re


def parse_tool_output(output):
    
    tool_pattern = r"<action>\{(.*?)\}</action>"
    matches = re.findall(tool_pattern, output, re.DOTALL)
    parsed_tools = []
    for match in matches:
        try:
            # tool_data = json.loads("{" + match.strip() + "}")
            tool_data = json.loads(match.strip())
            parsed_tools.append(tool_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return parsed_tools

def extract_json(select_response):
    select_response = select_response.replace('```json', '').replace('```', '')
    start_index = select_response.find('{')
    end_index = select_response.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_str = select_response[start_index:end_index + 1]
        return json.loads(json_str)
    else:
        return json.loads(select_response)