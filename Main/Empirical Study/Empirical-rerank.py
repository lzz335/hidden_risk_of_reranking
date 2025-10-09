from RAG.ReRanker import RerankerByRerankerModel, OurReranker, combine_documents
import torch
from RAG.Config import TrainConfig
from tqdm import tqdm
from util.json_method import read_jsonl,write_dict_to_jsonl


model_name_list = ["BAAI/bge-reranker-v2-m3", "jinaai/jina-reranker-v2-base-multilingual", "Alibaba-NLP/gte-reranker-modernbert-base"]
device = "cuda" if torch.cuda.is_available() else "cpu"
from sentence_transformers import CrossEncoder
model = CrossEncoder('ByteDance/ListConRanker', trust_remote_code=True)
def get_rerank_data(recall_data,file_path):
    for item in tqdm(recall_data):
        query = item["query"]
        documents = item["document"]
        batch = [
            [query,documents[i]] for i in range(len(documents))
        ]
        score_list_wise = model.predict(batch).tolist()
        save_data = {
            "query": item["query"],
            "list_corpus": combine_documents(documents=item["document"], scores=score_list_wise),
            "answer": item["answer"],
            "list_score": score_list_wise,
            "documents": item["document"],
        }
        write_dict_to_jsonl(save_data, file_path)
rerank_dataset = read_jsonl("empirical/jina_mips_result.jsonl")
get_rerank_data(rerank_dataset, "empirical/jina_rerank_results_listwise.jsonl")
rerank_dataset = read_jsonl("empirical/qwen_mips_result.jsonl")
get_rerank_data(rerank_dataset, "empirical/qwen_rerank_results_listwise.jsonl")
