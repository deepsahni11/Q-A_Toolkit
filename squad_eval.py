import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(prediction_tokens)
    recall =  num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)




def evaluate(ground_truths, predictions):
    f1 = exact_match = total = 0
    for i, p in enumerate(predictions):
      total += 1
      ground_truth = ground_truths[i]
      exact_match += metric_max_over_ground_truths(
                exact_match_score, p, ground_truth)
      f1 += metric_max_over_ground_truths(
                f1_score, p, ground_truth)

    exact_match = 100.0 * exact_match /total
    f1 = 100.0 * f1 / total
    if f1>1:
        f1=f1/10

    return exact_match, f1
