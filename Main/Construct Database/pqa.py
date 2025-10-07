import json
from FlagEmbedding import FlagModel
import torch
import chromadb
from tqdm import tqdm
import os
chroma_client = chromadb.PersistentClient(path=r"../database")
chroma_client.delete_collection("MedPQA")
collection = chroma_client.create_collection(name="MedPQA")
batch_size = 16
index = 0
def get_embedding_and_meta_data_and_document(json_data, input_model, need_device = False):
    document = []
    meta_datas = []
    for data in json_data:
        document.append(data["paragraph_text"])
        meta_datas.append({
            "id": str(data["id"]),
        })
    if not need_device:
        embeddings = input_model.encode(document)
    else:
        embeddings = input_model.encode(document, task="text-matching")

    return document, meta_datas, embeddings


embeddings_model = FlagModel('BAAI/bge-large-en-v1.5',
                             query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                             use_fp16=True,
                  )
full_context = []
with open("../dataset/pqa/full_dataset.json", "r", encoding="utf-8") as f:
    full = json.load(f)
for item in full:
    for dum in item["CONTEXTS"]:
        full_context.append({
            "id": item["ids"],
            "paragraph_text":dum
        })

for i in tqdm(range(0, len(full_context), batch_size)):
    with torch.no_grad():
        batch_json = full_context[i:min(i+batch_size,len(full_context))]
        documents, meta_datas, embeddings = get_embedding_and_meta_data_and_document(batch_json, embeddings_model)
        collection.add(
                ids=[str(ids) for ids in list(range(i,min(i+batch_size,len(full_context))))],
                documents=documents,
                embeddings=embeddings,
                metadatas=meta_datas
            )