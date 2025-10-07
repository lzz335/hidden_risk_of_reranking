from FlagEmbedding import FlagModel
import chromadb
from tqdm import tqdm
import torch
from util.json_method import read_jsonl

model = FlagModel('BAAI/bge-large-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True,
                  )
print(torch.cuda.is_available())
chroma_client = chromadb.PersistentClient(path="../database")
chroma_client.delete_collection("nq_full")
collection = chroma_client.create_collection(name="nq_full")

data = read_jsonl("dataset/train_dataset_2.jsonl")
data2 = read_jsonl("dataset/dev_dataset.jsonl")
full_data = data2 + data
batch_size = 64
index = 0
for i in tqdm(range(0, len(full_data), batch_size)):
    split_data = data[i:min(i + batch_size, len(full_data))]
    batch_documents = [item["documents"] for item in split_data]
    batch_metas = [item["metas_data"] for item in split_data]
    embeddings = model.encode(batch_documents)
    index += 1
    collection.add(
        # ids=["id{}".format(index)],
        ids=[str(ids) for ids in list(range(i, min(i + batch_size, len(full_data))))],
        documents=batch_documents,
        embeddings=embeddings,
        metadatas=batch_metas
    )