import numpy as np

from util.json_method import read_json_file,read_jsonl
corpus_data = read_json_file("Train Bert/train_corpus2.json")

data_dict = {}
for item in corpus_data:
    data_dict[item["sentence1"]+item["sentence2"]] = item["label"]
def combine_documents(documents, scores=None, top_k=5):
    if scores is None or type(scores[0]) is str:
        scores = [-index / len(documents) for index, _ in enumerate(documents)]
    if len(scores) == 3:
        scores = scores[0]
    assert len(documents) == len(scores)
    assert top_k <= len(documents)
    
    corpus = []
    for doc, score in zip(documents, scores):
        corpus.append({"document": doc, "score": score})
    new_corpus = sorted(corpus, key=lambda x: x["score"], reverse=True)
    return [new_corpus[i]["document"] for i in range(top_k)]
def get_number(ids, dataset):
    count = [0,0,0,0,0]
    sum_number = [0,0,0,0,0]
    for item in dataset:
        for j in range(5):
            if item["query"] + combine_documents(item["documents"])[j] in data_dict.keys():
                sum_number[j] += 1
                if data_dict[item["query"] + combine_documents(item["documents"])[j]] == ids:
                    count[j] += 1
    return [count[i]/sum_number[i] for i in range(5)]
data2 = read_jsonl("empirical/qwen_rerank_results.jsonl")
data = read_jsonl("empirical/jina_rerank_results.jsonl")


def get_number(name, ids, dataset):
    count = [0,0,0,0,0]
    sum_number = [0,0,0,0,0]
    for item in dataset:
        for j in range(5):
            if item["query"] + combine_documents(item["documents"], item[name])[j] in data_dict.keys():
                if data_dict[item["query"] + combine_documents(item["documents"], item[name])[j]] == ids:
                    count[j] += 1
                sum_number[j] += 1
    return [count[i] for i in range(5)]
types = ["query","bge_score", "jina_score", "gte_score", "our_score"]
res = []
for ts in types:
    res.append(get_number(ts,1,data2))
np.savetxt("empirical/qwen_score.csv", np.array(res), delimiter=",")
res = []
for ts in types:
    res.append(get_number(ts,1,data))
np.savetxt("empirical/jina_score.csv", np.array(res), delimiter=",")