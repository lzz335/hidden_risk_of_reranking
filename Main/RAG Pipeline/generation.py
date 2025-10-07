# Import necessary libraries for API calls, concurrent processing, and data handling
import json
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import torch  # Import torch for logarithm operations

from RAG.ReRanker import combine_documents
from util.json_method import read_jsonl


# Template for prompting language models with retrieved context
query_format = """Context (which may or may not be relevant):
<Retrieved documents>
Question: <Question>
Answer:
"""
# Initialize OpenAI client for GPT models (commercial API)
client1 = OpenAI(
        base_url="https://api.key77qiqi.cn/v1",
        api_key="sk-WtnyAr3AC4lV6ip1FwBFZVGAgieqRySdAu7hME3FKRCQT9TG"
)

# Initialize OpenAI client for local models via Ollama
client2 = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible interface
    api_key="ollama"  # Dummy key for local inference
)

def get_one_answer(query: str, question_type, corpus, q_id, model_name):
    """Generate answer for a single query using specified LLM and context
    
    This function handles API calls to different language models (GPT via API
    or local models via Ollama) with retry logic and specialized prompts.
    
    Args:
        query: The question to answer
        question_type: Type of processing ('none' or re-ranking method)
        corpus: Retrieved context documents
        q_id: Unique question identifier
        model_name: Name of the language model to use
        
    Returns:
        Dictionary containing q_id and answer (plus reasoning for some models)
    """
    # Select appropriate client based on model type
    global client1
    global client2
    client = client1 if "gpt" in model_name else client2
    count = 0
    while True:  # Retry loop for API reliability
        count += 1
        try:
            # Select system prompt based on question type
            if question_type == "none":
                # Direct QA without context
                content = "You are an expert in question answering. Your answer only needs to be \"yes\" or \"no\" or \"maybe\", without any sentence description. Your answer needs to be returned in JSON format, for example: {\"answer\": \"Your answer\"}.
            else:
                content = """Consider the context provided carefully, as some documents may contain "supportive noise"—information that is semantically relevant but does not provide the exact answer needed. Your task is to accurately discern the "gold reference" from any "supportive evidence" and base your response on the most reliable information available. Avoid being misled by supportive noise and focus on identifying the precise information required to answer the question correctly, and avoid being misled by content that appear semantically relevant but lacks logical causality. You must provide an answer in the value of the "Your answer", even if it is not included in the corpus. Your answer only needs to be \"yes\" or \"no\" or \"maybe\", without any sentence description. Your answer needs to be returned in JSON format, for example: {\"answer\": \"Your answer\"}."""
            # Check token limit to avoid API errors
            if len(query_format.replace(
                    "<Retrieved documents>", corpus).replace(
                "<Question>", query)) > 128000:
                # Return unknown if context exceeds token limit
                return {
                    "q_id": q_id,
                    "answer": "unknown",
                }
            else:
                # Create API request with context and question
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.1,  # Low temperature for consistent answers
                    messages=[
                        {
                            "role": "system",
                            "content": content,
                        },
                        {
                            "role": "user",
                            "content": query_format.replace(
                                "<Retrieved documents>", corpus).replace(
                                "<Question>", query) if question_type != "none"
                            else "Question: <Question>\nAnswer:".replace("<Question>", query),
                        },
                    ],
                    max_tokens=1024
                )
                # Clean response text and extract answer
                answer = response.choices[0].message.content.replace("```json", "").replace("```", "").replace("\n", "")
                # Handle different response formats for different models
                if "qwen" not in model_name:
                    # Standard JSON response handling (GPT models)
                    if answer == "yes" or answer =="no" or answer == "maybe":
                        result = {
                            "q_id": q_id,
                            "answer": answer
                        }
                    else:
                        # Parse JSON response for more complex answers
                        result = {
                            "q_id": q_id,
                            "answer": json.loads(answer)["answer"]
                        }
                else:
                    reasoning_content = answer.split("<think>")[1].split("</think>")[0]
                    final_answer = json.loads(answer.split("</think>")[1].replace("\n", ""))
                    result = {
                        "q_id": q_id,
                        "answer": final_answer["answer"],
                        "reasoning_content": reasoning_content,
                    }
                return result
        except Exception as e:
            print(q_id, e)
            if count == 1:
                result = {
                    "q_id": q_id,
                    "answer": "unknown",
                    "reasoning_content": "unknown"
                }
                return result


def multi_query(queries: List[Dict], generation_method, reranker, base_model: str, limit_thread=100, ablation_study=False, ablation_type=None):
    assert not (ablation_type and ablation_study is None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=limit_thread) as executor:
        futures = {}
        for i in range(len(queries)):
            if not ablation_study:
                if reranker != "none":
                    future = executor.submit(
                        generation_method,
                        queries[i]["query"],
                        reranker,
                        queries[i][reranker],
                        i,
                        base_model
                    )
                else:
                    future = executor.submit(
                        generation_method,
                        queries[i]["query"],
                        reranker,
                        "",
                        i,
                        base_model
                    )
                futures[future] = i
            else:
                future = executor.submit(
                    generation_method,
                    queries[i]["query"],
                    reranker,
                    combine_documents(queries[i]["documents"], score_implement(queries[i]["our_score"], ablation_type)),
                    i,
                    base_model
                )
                futures[future] = i

        result = [None] * len(queries)

        with tqdm(total=len(queries), desc=f"Processing queries, model: {base_model}\t, baseline: {reranker}\t") as pbar:
            for future in concurrent.futures.as_completed(futures):
                original_index = futures[future]
                result[original_index] = future.result()
                pbar.update(1)

    return result

def score_implement(scores, type_name):
    weight = [1, -0.5]
    _, p1, p2 = scores
    if type_name == "-md":
        score = [torch.log(p2[i][1] if p2[i][1] > 0.5 else 0.5) * weight[1] for i in range(len(p1))]
    elif type_name == "-clip":
        score = [torch.log(p1[i][1]) * weight[0] + torch.log(p2[i][1]) * weight[1] for i in range(len(p1))]
    elif type_name == "-hd":
        score = [torch.log(p1[i][1]) * weight[0] for i in range(len(p1))]
    else:
        return [-index / 50 for index in range(50)]
    return score


if __name__ == '__main__':
    # For PQA

    dataset_name = "PQA"
    data = read_jsonl("dataset/rerank_results_{}.jsonl".format(dataset_name))

    model_list = [
        "gemma3:4b",
        "gpt-4o-mini",
        "qwen3:8b",
    ]
    # question_types = ["bge_corpus", "gte_corpus", "list_corpus", "our_corpus", "jina_corpus", "mips_corpus", "none"]
    # for model_name in model_list:
    #     for question_type in question_types:
    #         recall_data = multi_query(data, get_one_answer, question_type,model_name)
    #         sort_data = sorted(recall_data, key=lambda x: x["q_id"], reverse=False)
    #         with open("result/PQA/recall_index_{}_{}_{}.json".format(model_name,dataset_name, question_type), "w", encoding="utf8") as f:
    #             json.dump(sort_data, f, indent=2)
    #         print(f"Task for model: {model_name}, dataset: {dataset_name}, method: {question_type} have finished!")
    ablation_types = ["-md", "-clip", "-hd"]
    for model_name in model_list:
        for ablation_type in ablation_types:
            recall_data = multi_query(data, get_one_answer, "our_corpus", model_name, ablation_study=True, ablation_type=ablation_type)
            sort_data = sorted(recall_data, key=lambda x: x["q_id"], reverse=False)
            with open("result/PQA/recall_index_{}_{}_{}.json".format(model_name, dataset_name, ablation_type), "w", encoding="utf8") as f:
                json.dump(sort_data, f, indent=2)
            print(f"Task for model: {model_name}, dataset: {dataset_name}, ablation_type: {ablation_type} have finished!")