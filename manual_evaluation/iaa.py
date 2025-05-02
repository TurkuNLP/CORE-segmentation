import json
import numpy as np
from sklearn.metrics import cohen_kappa_score


def read_scores_file(filename):
    scores = {"label": [], "segment": []}
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            scores["label"].append(data["label_score"])
            scores["segment"].append(data["segment_score"])
    return scores


def read_samples_file(filename):
    """Read the original samples file to identify entries with single segments"""
    single_segment_indices = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            if len(data["segments"]) == 1:
                single_segment_indices.append(i)
    return single_segment_indices


def quadratic_weighted_kappa(rater1, rater2):
    # Convert to numpy arrays
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)

    return cohen_kappa_score(rater1, rater2, weights="quadratic")


def print_stats(scores1, scores2, name):
    """Print descriptive statistics for a score type"""
    s1 = np.array(scores1)
    s2 = np.array(scores2)

    print(f"\n{name} Statistics:")
    print(
        f"Annotator 1 - Mean: {np.mean(s1):.2f}, Std: {np.std(s1):.2f}, Min: {np.min(s1)}, Max: {np.max(s1)}"
    )
    print(
        f"Annotator 2 - Mean: {np.mean(s2):.2f}, Std: {np.std(s2):.2f}, Min: {np.min(s2)}, Max: {np.max(s2)}"
    )
    print(
        f"Score distribution Annotator 1:",
        dict(zip(*np.unique(s1, return_counts=True))),
    )
    print(
        f"Score distribution Annotator 2:",
        dict(zip(*np.unique(s2, return_counts=True))),
    )


def calculate_iaa(file1, file2, samples_file=None, filter_single_segment=False):
    # Read both files
    scores1 = read_scores_file(file1)
    scores2 = read_scores_file(file2)

    # Take only the minimum length to ensure we compare same number of items
    min_len = min(len(scores1["label"]), len(scores2["label"]))

    # If filtering for single-segment texts
    if filter_single_segment and samples_file:
        single_segment_indices = read_samples_file(samples_file)
        print(f"\nTotal single-segment texts found: {len(single_segment_indices)}")

        # Filter scores to only include single-segment texts
        filtered_scores1 = {
            "label": [
                scores1["label"][i] for i in single_segment_indices if i < min_len
            ],
            "segment": [
                scores1["segment"][i] for i in single_segment_indices if i < min_len
            ],
        }

        filtered_scores2 = {
            "label": [
                scores2["label"][i] for i in single_segment_indices if i < min_len
            ],
            "segment": [
                scores2["segment"][i] for i in single_segment_indices if i < min_len
            ],
        }

        # Update variables for the rest of the function
        scores1 = filtered_scores1
        scores2 = filtered_scores2
        min_len = len(filtered_scores1["label"])
        print(f"Total single-segment items compared: {min_len}")
    else:
        print(f"\nTotal items compared: {min_len}")

    # Print segment by segment comparison
    print("\nSegment by segment comparison (for label scores):")
    for i, (s1, s2) in enumerate(zip(scores1["label"], scores2["label"])):
        print(f"Item {i}: {s1} vs {s2}")

    # Print statistics for both types of scores
    print_stats(scores1["label"], scores2["label"], "Label Scores")
    print_stats(scores1["segment"], scores2["segment"], "Segment Scores")

    # Calculate weighted kappa for both types of scores
    label_kappa = quadratic_weighted_kappa(scores1["label"], scores2["label"])
    segment_kappa = quadratic_weighted_kappa(scores1["segment"], scores2["segment"])

    return {"label_score_iaa": label_kappa, "segment_score_iaa": segment_kappa}


# Usage
if __name__ == "__main__":
    # Calculate IAA for all texts
    print("\n=== IAA FOR ALL TEXTS ===")
    results_all = calculate_iaa("evaluations_A.jsonl", "evaluations_B.jsonl")
    print(f"\nFinal IAA Scores (All Texts):")
    print(f"Label Score IAA: {results_all['label_score_iaa']:.3f}")
    print(f"Segment Score IAA: {results_all['segment_score_iaa']:.3f}")

    # Calculate IAA for single-segment texts only
    print("\n=== IAA FOR SINGLE-SEGMENT TEXTS ONLY ===")
    results_single = calculate_iaa(
        "evaluations_A.jsonl",
        "evaluations_B.jsonl",
        "sample.jsonl",
        filter_single_segment=True,
    )
    print(f"\nFinal IAA Scores (Single-Segment Texts Only):")
    print(f"Label Score IAA: {results_single['label_score_iaa']:.3f}")
    print(f"Segment Score IAA: {results_single['segment_score_iaa']:.3f}")
