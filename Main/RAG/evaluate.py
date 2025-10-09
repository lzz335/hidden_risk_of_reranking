# Import utilities for reading JSON and JSONL files
from util.json_method import read_json_file, read_jsonl

def get_file_path(model_name, dataset_name, question_type):
    """Generate standardized file path for model results
    
    Args:
        model_name: Name of the model (e.g., 'gpt-4o-mini')
        question_type: Type of question processing (e.g., 'none', 'our_corpus')
        
    Returns:
        Formatted file path for the model results
    """
    return "result/PQA/recall_index_{}_{}_{}.json".format(model_name, dataset_name, question_type)


def calculate_metrics(gold_answers, predictions):
    """Calculate evaluation metrics for yes/no/maybe classification
    
    Computes special accuracy (treating 'maybe' as correct for 'yes'),
    macro F1, and micro F1 scores.
    
    Args:
        gold_answers: List of ground truth answers
        predictions: List of model predictions
        
    Returns:
        Tuple of (special_accuracy, macro_f1, micro_f1)
    """
    # Define valid classes for classification
    classes = ["yes", "no", "maybe"]

    # Normalize both gold and predicted answers
    gold = [ans.strip().lower() for ans in gold_answers]
    pred = [ans1.strip().lower() for ans1, ans2 in zip(predictions, gold)]

    # Validate input lengths
    if len(gold) != len(pred):
        raise ValueError("Gold answers and predictions must have the same length")

    correct = 0
    for g, p in zip(gold, pred):
        if g == p:
            correct += 1
    accuracy_special = correct / len(gold)

    # Build confusion matrix
    conf_matrix = [[0] * 3 for _ in range(3)]
    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for g, p in zip(gold, pred):
        if g in label_to_idx and p in label_to_idx:
            conf_matrix[label_to_idx[g]][label_to_idx[p]] += 1

    # Calculate F1 scores for each class
    f1s = []
    for c in range(3):
        tp = conf_matrix[c][c]  # True positives
        fp = sum(row[c] for row in conf_matrix) - tp  # False positives
        fn = sum(conf_matrix[c]) - tp  # False negatives

        # Calculate precision, recall, and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)

    # Macro F1: average of F1 scores across all classes
    macro_f1 = sum(f1s) / 3

    # Micro F1: overall accuracy
    correct_standard = sum(conf_matrix[i][i] for i in range(3))
    micro_f1 = correct_standard / len(gold)

    return accuracy_special, macro_f1, micro_f1

def calculate_pipline(model_name, dataset_name, question_type):
    """Evaluate pipeline performance for a specific model and question type
    
    Args:
        model_name: Name of the model being evaluated
        question_type: Type of question processing method
    """
    # Get file path for model results
    path = get_file_path(model_name, dataset_name, question_type)
    # Load model predictions
    recall_data = read_json_file(path)
    # Extract answer predictions
    recall_result = [item["answer"] for item in recall_data]
    # Calculate and print metrics
    print(f"{model_name}\t, {question_type}\t:", calculate_metrics(gold_answer, recall_result))
# Load RAG dataset and extract ground truth answers
rag_dataset = read_jsonl("dataset/mips_recall_results_PQA.jsonl")
gold_answer = [item["answer"] for item in rag_dataset]

# List of models to evaluate
model_list = [
    "gpt-4o-mini",    # OpenAI GPT-4o Mini
    "qwen3:8b",       # Qwen 3 8B parameter model
    "gemma3:4b"       # Gemma 3 4B parameter model
]

# List of question processing types to evaluate
# none: No processing
# *_corpus: Different retrieval methods
# our_corpus: Our proposed method
# -*: Ablation study variants
question_types = [
    "none", "mips_corpus", "bge_corpus", "jina_corpus", 
    "gte_corpus", "list_corpus", "our_corpus", 
    "-clip", "-hd", "-md"
]

# Run evaluation for all model and question type combinations
for model_name in model_list:
    for question_type in question_types:
        calculate_pipline(model_name, "PQA", question_type)