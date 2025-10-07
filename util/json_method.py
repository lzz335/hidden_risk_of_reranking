import json
import os


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            try:
                # print(index)
                item = json.loads(line.strip())
                data.append(item)
            except Exception as e:
                print(f"Error processing line {index}: {e}")
    return data


def write_dict_to_jsonl(data_dict: dict, file_path: str):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass
    with open(file_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(data_dict, ensure_ascii=False)
        f.write(json_line + '\n')


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data
