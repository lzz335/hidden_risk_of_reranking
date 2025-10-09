from util.json_method import read_json_file, read_jsonl
import chromadb
from RAG.MIPS import MIPS, save_recall_results
from FlagEmbedding import FlagModel
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import chromadb
from RAG.MIPS import MIPS, save_recall_results
import torch

pre_dataset = read_jsonl("dataset/musique_ans_v1.0_dev.jsonl")
dataset = []
for dum in pre_dataset:
    if dum["answerable"] is True:
        dataset.append(dum)

chroma_client = chromadb.PersistentClient(path=r"../database")
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_mips_result(multi_qa_dataset:List[Dict], mips_method, path, question_key ="question",answer_key = "answer",keys = None):
    questions = [qa[question_key] for qa in multi_qa_dataset]
    answers = [qa[answer_key] for qa in multi_qa_dataset]
    meta_results, documents = mips_method.query(questions)
    save_recall_results(query_text=questions, documents=documents, metadatas= meta_results, jsonl_path=path,keys=keys, answers=answers)

embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(device)
collection = chroma_client.get_collection("db_musique_jina_base")
mips_model = MIPS(embedding_model, collection)
mips_model.task = 1
get_mips_result(dataset,mips_model,path="empirical/jina_mips_result.jsonl",keys=[])

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
collection = chroma_client.get_collection("db_musique_qwen_base")
mips_model = MIPS(embedding_model, collection)
mips_model.task = 1
get_mips_result(dataset,mips_model,path="empirical/qwen_mips_result.jsonl",keys=[])