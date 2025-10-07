import json
import os.path
from Prompt import query_format
import tqdm

from typing import List, Dict
import concurrent.futures

from openai import OpenAI


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file_path:
        json_data = json.load(json_file_path)
        return json_data


client = OpenAI(
    base_url="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)


def filter_data(answer):
    try:
        return json.loads(answer)
    except ValueError as e:
        print(e)
        # print(answer)
    return {}


def generate_response(question_input):
    query_input = (query_format.
                   replace("<Sentence A>", question_input["sentence1"]).
                   replace("<Sentence B>", question_input["sentence2"]))
    while True:
        try:
            completion_instance = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": query_input}],
                temperature=0.5,
                top_p=1,
                response_format={"type": "json_object"},
                max_tokens=1024
            )
            # print(completion_instance.choices[0].message.content)
        except Exception as e:
            print(e)
            # print(question_input)
            continue
        if filter_data(completion_instance.choices[0].message.content) != {}:
            json_data = filter_data(completion_instance.choices[0].message.content)
            # print(json_data)
            json_data = json_data["result"]
            # print(json_data)
            if question_input["idx"] % 100 == 0:
                print("Question index:{} is finished!".format(question_input["idx"]))
            return {
                "idx": question_input["idx"],
                "is_supporting": question_input["idx"],
                "sentence1": question_input["sentence1"],
                "sentence2": question_input["sentence2"],
                "CN": json_data["CN"],
                "ISN": json_data["ISN"],
                "SeN": json_data["SeN"],
                "SuN": json_data["SuN"],
                "TR": json_data["TR"],
            }
        else:
            print(question_input)


def multi_query(queries: List[Dict], generation_method, limit_thread=40):
    with concurrent.futures.ThreadPoolExecutor(max_workers=limit_thread) as executor:
        futures = [executor.submit(generation_method, queries[i]) for i in range(len(queries))]
    result = []
    for future in concurrent.futures.as_completed(futures):
        result.append(future.result())
    return result


query_data = read_json_file("data/musique.json")
batch_number = 1000
split_number = len(query_data) // batch_number + 1
for i in tqdm.tqdm(range(0, split_number)):
    res = multi_query(query_data[batch_number * i:min(batch_number * (i + 1), len(query_data))], generate_response,
                      limit_thread=100)
    with open("data/musique/train_filter_{}.json".format(i), "w") as f:
        json.dump(res, f, indent=2)
