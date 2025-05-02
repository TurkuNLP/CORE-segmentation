import json
from cleanlab import Datalab
import numpy as np


def load_and_process_jsonl(file_path):
    with open(file_path, "r") as f:
        records = [json.loads(line) for line in f]

    index_labels = []
    pred_probs = []
    texts = []
    labels_str = []

    for record in records:
        labels = [i for i, val in enumerate(record["labels"]) if val == 1]
        index_labels.append(labels)
        pred_probs.append(record["pred_probs"])
        texts.append(record["text"])
        labels_str.append(record["labels_str"])

    return index_labels, np.array(pred_probs), texts, labels_str


def filter_data(file_path, output_path, bad_output_path):
    labels, pred_probs, texts, labels_str = load_and_process_jsonl(file_path)

    lab = Datalab(
        data={"labels": labels, "text": texts},
        label_name="labels",
        task="multilabel",
    )

    # Get label issues with scores
    lab.find_issues(pred_probs=pred_probs)
    label_issues = lab.get_issues("label")
    issue_df = label_issues.query("is_label_issue")

    # Get sorted indices of problematic examples (worst first)
    sorted_issues = issue_df.sort_values("label_score").index.values

    # Create mask for keeping examples
    keep_indices = np.ones(len(texts), dtype=bool)
    keep_indices[issue_df.index.values] = False

    # Filter and write good data
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (label, text) in enumerate(zip(labels_str, texts)):
            if keep_indices[i]:
                f.write(f"{label}\t{text}\n")

    # Write bad examples in order of increasing label_score (worst first)
    with open(bad_output_path, "w", encoding="utf-8") as f:
        for i in sorted_issues:
            f.write(f"{labels_str[i]}\t{texts[i]}\n")

    return len(sorted_issues)  # Return number of removed examples


if __name__ == "__main__":
    file_path = "test_predictions_filtered.jsonl"
    output_path = "en_core_filtered_cleanlab.tsv"
    bad_output_path = "en_core_filtered_bad.tsv"
    removed_count = filter_data(file_path, output_path, bad_output_path)
    print(f"Removed {removed_count} examples")
