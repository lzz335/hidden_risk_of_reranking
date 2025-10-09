import json
import os

import numpy as np
import scipy.stats as stats
import re


def lcs_length(reference, generated):
    """
    Calculate the length of the longest common subsequence (LCS) between two sequences.
    
    Args:
        reference: The reference sequence
        generated: The generated sequence
    
    Returns:
        int: Length of the longest common subsequence
    """
    # Create a 2D array to store LCS lengths
    m = len(reference)
    n = len(generated)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Dynamic programming to calculate LCS length
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == generated[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def calculate_one_rouge_l(reference, generated):
    generated = re.sub(r'[^\w\s]', '', generated)
    reference = re.sub(r'[^\w\s]', '', reference)
    reference_tokens = reference.strip().lower().split()
    generated_tokens = generated.strip().lower().split()

    lcs_len = lcs_length(reference_tokens, generated_tokens)

    recall = lcs_len / len(reference_tokens) if reference_tokens else 0
    precision = lcs_len / len(generated_tokens) if generated_tokens else 0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1
    }


def rouge_l(generated, reference):
    if type(generated) is dict:
        generated = list(generated.values())
    if type(generated) is list:
        try:
            generated = " ".join(generated)
        except Exception as e:
            generated = str(generated)
    if type(reference) is list:
        max_recall = 0
        max_f1 = 0
        max_precision = 0
        for ref in reference:
            res = calculate_one_rouge_l(ref, generated)
            if res['f1'] > max_f1:
                max_f1 = res['f1']
            if res['recall'] > max_recall:
                max_recall = res['recall']
            if res['precision'] > max_precision:
                max_precision = res['precision']

        return {
            'recall': max_recall,
            'precision': max_precision,
            'f1': max_f1
        }
    else:
        return calculate_one_rouge_l(reference, generated)


def f1_one_score(prediction, ground_truth):
    """
    Calculate F1 score between prediction and ground truth.
    
    Args:
        prediction: Predicted text string
        ground_truth: Ground truth text string
    
    Returns:
        float: F1 score
    """
    # Convert input to lowercase and split into word lists
    prediction_tokens = set(prediction.lower().split())
    ground_truth_tokens = set(ground_truth.lower().split())

    # Calculate intersection size
    intersection = prediction_tokens.intersection(ground_truth_tokens)

    # Calculate TP, FP, FN
    TP = len(intersection)
    FP = len(prediction_tokens) - TP
    FN = len(ground_truth_tokens) - TP

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def f1_score(prediction, ground_truth):
    """
    Calculate F1 score between prediction and ground truth.
    
    Args:
        prediction: Predicted text (string, list, or dict)
        ground_truth: Ground truth text (string, list, or dict)
    
    Returns:
        float: Maximum F1 score across all ground truth references
    """
    if type(prediction) is dict:
        prediction = list(prediction.values())
    if type(prediction) is list:
        try:
            prediction = " ".join(prediction)
        except Exception as e:
            prediction = str(prediction)
    if type(ground_truth) is list:
        max_f1 = 0
        for gt in ground_truth:
            if f1_one_score(prediction, gt) > max_f1:
                max_f1 = f1_one_score(prediction, gt)
        return max_f1
    else:
        return f1_one_score(prediction, ground_truth)


def preprocess(text):
    """
    Preprocess text: remove leading/trailing whitespace, convert to lowercase, remove punctuation.
    
    Args:
        text: Input text string
    
    Returns:
        str: Preprocessed text
    """
    text = text.strip().lower()  # Remove leading/trailing whitespace and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def exact_match_score(prediction, ground_truths):
    """
    Calculate Exact Match score.

    Args:
        prediction (str): Model-generated answer.
        ground_truths (list or str): Reference answers, can be a single answer or list of multiple answers.

    Returns:
        bool: Whether there is an exact match.
    """
    if type(prediction) is dict:
        prediction = list(prediction.values())
    if type(prediction) is list:
        try:
            prediction = " ".join(prediction)
        except Exception as e:
            prediction = str(prediction)
    # If ground_truths is a single answer, convert it to a list
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    # Preprocess the predicted answer
    prediction = preprocess(prediction)

    # Iterate through all reference answers, check for any match
    for gt in ground_truths:
        gt = preprocess(gt)
        if prediction == gt:
            return True  # Return True as soon as one match is found

    return False  # Return False if no match is found


def analyze_significance(method1, method2, alpha=0.05):
    """
    Analyze significance differences between two independent samples and report statistical tests and effect sizes.

    Args:
        method1: Data list for the first observation method
        method2: Data list for the second observation method
        alpha: Significance level (default 0.05)

    Output:
        Print test selection, p-value, effect size and conclusion
    """
    # Normality test (Shapiro-Wilk)
    norm1 = stats.shapiro(method1)
    norm2 = stats.shapiro(method2)

    # Homogeneity of variance test (Levene)
    var_test = stats.levene(method1, method2)

    # Initialize variables
    test_used = ''
    p_value = None
    effect_size = ''

    # Check if both samples follow normal distribution
    if norm1[1] > alpha and norm2[1] > alpha:
        print("Both samples satisfy normal distribution.")
        # Homogeneity of variance judgment
        if var_test[1] > alpha:
            # Equal variance, use Student t-test
            t_stat, p_value = stats.ttest_ind(method1, method2, equal_var=True)
            test_used = "Student t-test (equal variance)"
        else:
            # Unequal variance, use Welch t-test
            t_stat, p_value = stats.ttest_ind(method1, method2, equal_var=False)
            test_used = "Welch t-test (unequal variance)"

        # Calculate effect size Cohen's d
        n1, n2 = len(method1), len(method2)
        mean_diff = np.mean(method1) - np.mean(method2)
        var1, var2 = np.var(method1, ddof=1), np.var(method2, ddof=1)
        pooled_var = ((n1 * var1) + (n2 * var2)) / (n1 + n2)
        pooled_std = np.sqrt(pooled_var)
        d = mean_diff / (pooled_std * np.sqrt(1 / n1 + 1 / n2))
        effect_size = f'Cohen\'s d: {d:.2f}'
    else:
        print("At least one sample does not follow normal distribution, using non-parametric test.")
        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(method1, method2, alternative='two-sided')
        test_used = "Mann-Whitney U test"
        # Effect size r
        r = u_stat / (len(method1) * len(method2))
        effect_size = f'Effect size r: {r:.2f}'

    # Output results
    print(f'Test method used: {test_used}')
    print(f'p-value: {p_value:.4f}')
    print(f'Effect size: {effect_size}')

    if p_value < alpha:
        print("Conclusion: Statistically significant difference (p < 0.05)")
    else:
        print("Conclusion: No statistically significant difference (p ≥ 0.05)")


def write_dict_to_jsonl(data_dict: dict, file_path: str):
    """
    Write dictionary to specified JSONL file.

    Args:
        data_dict: Dictionary data to write
        file_path: Path to the JSONL file
    """
    # Check if file exists, create if not
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            print("File does not exist, creating file.")
            pass  # Create empty file

    # Write data in append mode
    with open(file_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(data_dict, ensure_ascii=False)
        f.write(json_line + '\n')
