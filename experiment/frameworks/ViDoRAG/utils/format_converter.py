import json
import os
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from llama_index.core import Document
from typing import Optional, List, Mapping, Any, Dict
def nodes2dict(nodes) -> List[Dict[str, Any]]:
    resp_dict = {
        "response": None,
        "source_nodes": [],
        "metadata": None
    }
    for node in nodes:
        # node.embedding = None
        resp_dict["source_nodes"].append(node.to_dict())
    return resp_dict

def nodefile2node(input_file):
    nodes = []
    for doc in json.load(open(input_file, 'r')):
        if doc['class_name'] == 'TextNode' and doc['text'] != '':
            nodes.append(TextNode.from_dict(doc))
        elif doc['class_name'] == 'ImageNode':
            nodes.append(ImageNode.from_dict(doc))
        else:
            continue
    return nodes

def onlchunkfile2node(input_file):
    content_json = json.load(open(input_file, 'r'))
    nodes = []
    for data in content_json:
        node = TextNode(text=data['title'] + data.get('hier_title', '') + data['content'], file_name=input_file)
        nodes.append(node)
        if len(nodes) > 1:
            nodes[-1].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[-2].node_id
            )
            nodes[-2].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[-1].node_id
            )
    return nodes


def transform_idp2markdown(response_json: dict) -> str:

    markdown_text = ""
    if 'layouts' in response_json:
        response_json = response_json['layouts']


    for layout in response_json:
        if layout is None:
            continue
        if not 'subType' in layout:
            layout['subType'] = 'para'

        if layout["type"] == "title": 

            markdown_text += "\n\n\n# " + layout["text"] + '\n'
        else:

            markdown_text += layout["text"] + "\n"
    return markdown_text

def documentfile2document(input_file):
    documents = []
    for doc in json.load(open(input_file, 'r')):
        document = Document.from_dict(doc)
        documents.append(document)
    return documents

def idpfile2document(input_file):
    content_json = json.load(open(input_file, 'r'))
    text = transform_idp2markdown(content_json)
    metadata = {"file_name": input_file}
    documents = [Document(text=text, metadata=metadata)]
    return documents

def text2document(input_file):
    text = open(input_file, 'r').read()
    metadata = {"file_name": input_file}
    documents = [Document(text=text, metadata=metadata)]
    return documents

def idpfile2text(input_file):
    content_json = json.load(open(input_file, 'r'))
    text = transform_idp2markdown(content_json)
    return text