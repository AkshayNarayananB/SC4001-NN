"""POPE metrics. Scoring follows the VCD reference (experiments/eval/eval_pope.py) exactly so
numbers are comparable with the paper: a gold-'yes' item is correct iff 'yes' appears in the
lowercased answer, a gold-'no' item iff 'no' appears."""
import re


def is_correct_reference(label: str, answer: str) -> bool:
    answer = answer.lower().strip()
    return ("yes" in answer) if label == "yes" else ("no" in answer)


def first_word(answer: str) -> str:
    match = re.match(r"\W*(\w+)", answer.lower())
    return match.group(1) if match else ""


def pope_metrics(records: list[dict]) -> dict:
    """records: dicts with 'label' ('yes'/'no') and 'answer' (generated text)."""
    tp = fn = tn = fp = 0
    for r in records:
        correct = is_correct_reference(r["label"], r["answer"])
        if r["label"] == "yes":
            tp, fn = tp + correct, fn + (not correct)
        else:
            tn, fp = tn + correct, fp + (not correct)
    n = len(records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / n,
        # answers whose first word is neither yes nor no (should be ~0 with the one-word prompt)
        "unclear_rate": sum(first_word(r["answer"]) not in ("yes", "no") for r in records) / n,
    }
