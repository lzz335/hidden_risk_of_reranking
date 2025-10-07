import json
from util.filter_nq_set import clean_intervals
from util.filter_nq_set import filter_nq_set_dev
from util.rag_util import write_dict_to_jsonl
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        index = 0
        for line in file:
            item = json.loads(line.strip())
            data.append(item)
            index += 1
    return data


def get_answer_document_meta_datas(json_data):
    word_list = json_data["document_text"].split(" ")
    question = json_data["question_text"]
    # print(json_data["annotations"])
    gold_answer = {}
    for short_test in json_data["annotations"]:
        if len(short_test["short_answers"])>0:
            short_answer_start = short_test["short_answers"][0]["start_token"]
            short_answer_end = short_test["short_answers"][0]["end_token"]
            temp_str = " ".join(word_list[short_answer_start:short_answer_end])
            gold_answer[temp_str] = temp_str
    if len(gold_answer) == 0:
        return None,None,None,None
    gold_answer = list(gold_answer.keys())
    documents = []
    metas_data = []
    for long_answer_candidate in json_data["long_answer_candidates"]:
        start_token_long = long_answer_candidate["start_token"]
        end_token_long = long_answer_candidate["end_token"]
        
        flag = False
        for annotation in json_data["annotations"]:
            if annotation["long_answer"]["start_token"] == start_token_long and annotation["long_answer"]["end_token"] == end_token_long:
                flag = True
                break
        metas_data.append(
            {
                "label": flag,
                "example_id": json_data["example_id"]
            }
        )
        if word_list[start_token_long] == "<P>":
            documents.append(" ".join(word_list[start_token_long+1:end_token_long-1]))
        else:
            documents.append(" ".join(word_list[start_token_long:end_token_long]))
    return question,gold_answer,documents,metas_data


def filter4dataset4jsonl(input_jsonl_path,output_jsonl_path_label,output_dataset_path,is_train=False):
    url_set = {""}
    index = 0
    with open(input_jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            item["long_answer_candidates"] = clean_intervals(item["long_answer_candidates"])
            query, gold_answer, documents, metas_data = get_answer_document_meta_datas(item)
            if gold_answer is None:
                continue
            set_label = item["document_url"] not in url_set
            for i in range(len(documents)):
                if set_label:
                    write_dict_to_jsonl(
                        {
                            "documents": documents[i],
                            "metas_data": metas_data[i]
                        },
                        output_dataset_path
                    )
                elif metas_data[i]["label"]:
                    write_dict_to_jsonl(
                        {
                            "documents": documents[i],
                            "metas_data": metas_data[i]
                        },
                        output_dataset_path
                    )
            if not is_train:
                write_dict_to_jsonl(
                    {
                        "query": query,
                        "gold_answer": gold_answer,
                        "example_id": item["example_id"]
                    }, 
                    output_jsonl_path_label
                )
                url_set.add(item["document_url"])
            index = index + 1
            if index % 1000 == 0:
                print(index)
            
filter4dataset4jsonl(r"../dataset/filter_dev_nq.jsonl","dataset/dev_label_2.jsonl","dataset/dev_dataset_2.jsonl")
filter4dataset4jsonl(r"../dataset/simplified-nq-train.jsonl","dataset/train_label.jsonl","dataset/train_dataset_2.jsonl",is_train=True)
