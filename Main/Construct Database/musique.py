from FlagEmbedding import FlagModel
import chromadb
import torch
from util.json_method import read_jsonl
import json
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
def get_embedding_and_meta_data_and_document(json_data, input_model, need_device = False):
    document = []
    meta_datas = []
    for data in json_data:
        document.append(data["paragraph_text"])
        meta_datas.append({
            "id": str(data["id"]),
            "title": data["title"],
            "is_supporting": data["is_supporting"],
        })
    if not need_device:
        embeddings = input_model.encode(document)
    else:
        embeddings = input_model.encode(document, task="text-matching")
    
    return document, meta_datas, embeddings


dataset = read_jsonl("dataset/musique_ans_v1.0_dev.jsonl")
dataset2 = read_jsonl("dataset/musique_ans_v1.0_train.jsonl")
dataset = dataset + dataset2
corpus = {}
for data in tqdm(dataset):
    for item in data["paragraphs"]:
        if item["paragraph_text"] not in corpus:
            corpus[item["paragraph_text"]] = {
                "is_supporting": item["is_supporting"],
                "paragraph_text": item["paragraph_text"],
                "title": item["title"],
                "id": [data["id"]] if item["is_supporting"] else [],
            }
        else:
            if item["is_supporting"]:
                if not corpus[item["paragraph_text"]]["is_supporting"]:
                    corpus[item["paragraph_text"]]["is_supporting"] = True
                corpus[item["paragraph_text"]]["id"].append(data["id"])
corpus_list = list(corpus.values())
with open("corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus_list, f, ensure_ascii=False, indent=4)


model = FlagModel('BAAI/bge-large-en-v1.5', 
                 query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                 use_fp16=True)
chroma_client = chromadb.PersistentClient(path="../database")
collection = chroma_client.create_collection(name="db_musique")
with open("corpus.json", "r", encoding="utf8") as f:
    corpus_list = json.load(f)
batch_size = 16
index = 0
for i in tqdm(range(0, len(corpus_list), batch_size)):
    with torch.no_grad():
        batch_json = corpus_list[i:min(i+batch_size,len(corpus_list))]
        documents, meta_datas, embeddings = get_embedding_and_meta_data_and_document(batch_json, model)
        collection.add(
                ids=[str(ids) for ids in list(range(i,min(i+batch_size,len(corpus_list))))],
                documents=documents,
                embeddings=embeddings,
                metadatas=meta_datas
            )