from util.json_method import read_json_file
import re
import chromadb
from tqdm import tqdm
from FlagEmbedding import FlagModel
corpus = read_json_file("dataset/corpus.json")
chroma_client = chromadb.PersistentClient(path=r"../database")
chroma_client.delete_collection("multihopQA")
collection = chroma_client.create_collection(name="multihopQA")
model = FlagModel('BAAI/bge-large-en-v1.5', 
                 query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                 use_fp16=True)
BATCH_SIZE = 64 
index = 0
batch_documents = []
batch_ids = []
batch_metadatas = []

for item in tqdm(corpus):
    texts = re.split(r'\n+', item["body"])
    for text in texts:
        if len(text) > 0:
            batch_documents.append(text)
            batch_ids.append(str(index))
            batch_metadatas.append({"title": item["title"], "category": item["category"]})
            index += 1
            
            if len(batch_documents) >= BATCH_SIZE:
                batch_embeddings = model.encode(batch_documents)
                
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                
                batch_documents = []
                batch_ids = []
                batch_metadatas = []

if len(batch_documents) > 0:
    batch_embeddings = model.encode(batch_documents)
    collection.add(
        ids=batch_ids,
        documents=batch_documents,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )
