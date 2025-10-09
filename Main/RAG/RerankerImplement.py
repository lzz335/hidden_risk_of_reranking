from transformers import AutoModelForSequenceClassification, AutoTokenizer
from RAG.Config import TrainConfig
import os
from sentence_transformers import CrossEncoder
from util.json_method import read_jsonl, write_dict_to_jsonl
from typing import List
from RAG.ReRanker import RerankerByRerankerModel, OurReranker, combine_documents
import torch
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
ListConRankerModel = CrossEncoder('ByteDance/ListConRanker', trust_remote_code=True)
model_name_list = ["BAAI/bge-reranker-v2-m3",
                   "jinaai/jina-reranker-v2-base-multilingual",
                   "Alibaba-NLP/gte-reranker-modernbert-base"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model_list = [AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device) for model_name in model_name_list]
reranker_model1 = RerankerByRerankerModel(model_list[0])
reranker_model1.set_tokenizer(AutoTokenizer.from_pretrained(model_name_list[0]))
reranker_model2 = RerankerByRerankerModel(model_list[1])
reranker_model3 = RerankerByRerankerModel(model_list[2])
reranker_model3.set_tokenizer(AutoTokenizer.from_pretrained(model_name_list[2]))
config2 = TrainConfig
config2.pretrained_model = "microsoft/deberta-base-mnli"
# 1， -0.5 is the setting in our paper
our_reranker = OurReranker(model_save_path=r"../model/full_nli.pt", device=device, inference_config=config2, weight=[1, -0.5])


def get_rerank_data(recall_data:List[dict],file_path,keys:None|List):
    for query in tqdm(recall_data):
        batch = [
            [query["query"],query["document"][i]] for i in range(len(query["document"]))
        ]
        score_list_wise = ListConRankerModel.predict(batch).tolist()
        score_bge = reranker_model1.reranker2(query["query"], query["document"])
        score_jina = reranker_model2.reranker(query["query"], query["document"])
        score_gte = reranker_model3.reranker2(query["query"], query["document"])
        score_our = our_reranker.rerank(query["query"], query["document"])
        save_data = {
            "query": query["query"],
            "bge_corpus": combine_documents(documents=query["document"], scores=score_bge),
            "jina_corpus": combine_documents(documents=query["document"], scores=score_jina),
            "gte_corpus": combine_documents(documents=query["document"], scores=score_gte),
            "our_corpus": combine_documents(documents=query["document"], scores=score_our[0]),
            "mips_corpus": combine_documents(documents=query["document"], scores=None),
            "list_corpus": combine_documents(documents=query["document"], scores=score_list_wise),
            "answer": query["answer"],
            "bge_score": score_bge,
            "jina_score": score_jina,
            "gte_score": score_gte,
            "list_score": score_list_wise,
            "our_score": score_our,
            "p1": score_our[1],
            "p2": score_our[2],
            "documents": query["document"],
        }
        if keys is not None:
            for key in keys:
                save_data[key] = query[key]
        write_dict_to_jsonl(save_data, file_path)
pqa_dataset = read_jsonl("dataset/mips_recall_results_PQA.jsonl")
get_rerank_data(pqa_dataset, "dataset/rerank_results_PQA.jsonl",keys=None)