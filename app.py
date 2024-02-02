


from flask import Flask, render_template, request, jsonify
import json
import os
from llama_index import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.schema import Document
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index import (VectorStoreIndex, get_response_synthesizer)
import openai 
import subprocess
import re 
import wandb
from weave.monitoring import StreamTable




def create_document_from_json_item(json_item):
    ti = json_item['title'] 
    des = json_item['description']
    keys = json_item['keywords']
    if keys: 
        ti += "keywords:" + keys
    document = Document(text=ti, metadata=json_item)
    return document

def generate_embeddings_for_document(document, model_name="BAAI/bge-small-en-v1.5"):
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    embeddings = embed_model.get_text_embedding(document.text)
    return embeddings


file_path = "./gpt4_menu_data_v2.json"
vec_index = None 

if not os.path.exists("./index"):        
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    documents = []
    for item in json_data:
        document = create_document_from_json_item(item)
        document_embeddings = generate_embeddings_for_document(document)
        document.embedding = document_embeddings
        documents.append(document)

    service_context = ServiceContext.from_defaults(llm=None, embed_model='local')
    vec_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    vec_index.storage_context.persist(persist_dir="./index")
else:
    storage_context = StorageContext.from_defaults(persist_dir="./index")
    service_context = ServiceContext.from_defaults(llm=None, embed_model='local')
    vec_index = load_index_from_storage(storage_context, service_context=service_context)




retriever = VectorIndexRetriever(index=vec_index, similarity_top_k=10)
service_context = ServiceContext.from_defaults(llm=None, embed_model='local')
response_synthesizer = get_response_synthesizer(service_context=service_context)
query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)])






# Login to W&B
wandb.login()
# Define constants for the StreamTable
WB_ENTITY = ""  # Set your W&B entity name here, or leave it empty to use the current logged-in entity
WB_PROJECT = "ai_menu"
STREAM_TABLE_NAME = "usage_data"


# Define a StreamTable
st = StreamTable(f"{WB_ENTITY}/{WB_PROJECT}/{STREAM_TABLE_NAME}")

app = Flask(__name__)

client = openai.OpenAI(api_key='sk-')




@app.route('/chat')
def index():
    return render_template('chat.html')

@app.route('/')
def menu():
    return render_template('index.html')




def parse_response_to_json(response_str):
    items = response_str.split("title: ")[1:]  # Split the response and ignore the first empty chunk
    json_list = []

    for item in items:
        lines = item.strip().split('\n')
        item_json = {
            "title": lines[0].strip(),
            "description": lines[1].replace("description: ", "").strip(),
            "keywords": lines[2].replace("keywords: ", "").strip(),
            "page": int(lines[3].replace("page: ", "").strip())
        }
        json_list.append(item_json)

    return json_list

def describe_items(json_list):
    description_str = "Some possible items you might be interested in include the following:<br><br>"
    for item in json_list:
        description_str += f"<strong>{item['title']}</strong> - {item['description']}<br><br>"

    return description_str


def query_index(query):
    response = query_engine.query(query)
    # Original results parsed to JSON
    return parse_response_to_json(str(response))
        



def generate_response_gpt(query, original_res):
    # Generating prompt for GPT
    prompt = f"This is a user at a restraunt searching for items to order. Given these initial results {original_res} for the following user query '{query}', return the JSON object for the items that make sense to include as a response (e.g., remove only items that are not at all relevant to the query='{query}') -- keep in mind that they may all be relevant and its perfectly fine to not remove any items. YOU MUST RETURN THE RESULT IN JSON FORM"
    # prompt = f"This is a user at a restraunt searching for items to order. Given the following user query and the search results {original_res} , return the JSON object for the items that make sense to include as a response (e.g., remove only items that you are sure the user wont be interested in given the query='{query}') -- keep in mind that they may all be relevant and its perfectly fine to not remove any items. Its ok to leave an item if it is only slightly relevant. If no items are to be removed, return an empty json object"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    if response.choices:
        reply = response.choices[0].message.content
        filtered_res_json_str = re.search(r"```json(.+?)```", reply, re.DOTALL)

        print(filtered_res_json_str)
        if filtered_res_json_str:
            filtered_res_json = json.loads(filtered_res_json_str.group(1))
            if not len(filtered_res_json): 
                return original_res
        else:
            filtered_res_json = original_res
        
        
        return filtered_res_json
    else:
        return original_res


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    original_res = query_index(query)
    filtered_res_json = generate_response_gpt(query, original_res)
    st.log({"query": query, "results": describe_items(filtered_res_json)})
    return jsonify({'res': describe_items(filtered_res_json)})
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)