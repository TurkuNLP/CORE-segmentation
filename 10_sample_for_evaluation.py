import json
import random
from collections import Counter
import statistics
import sys

# Set random seed
random.seed(1)


def sample_mixed_texts(input_file, output_file, n_samples=100, segmented_ratio=0.8):
    """
    Randomly sample texts from the input file with a specified ratio of segmented vs non-segmented texts.
    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        n_samples: Total number of samples wanted
        segmented_ratio: Ratio of texts with >1 segments (0.8 means 80% segmented, 20% non-segmented)
    """
    # Read all texts
    with open(input_file, "r", encoding="utf-8") as f:
        texts = [json.loads(line) for line in f]

    # Separate texts into segmented and non-segmented
    segmented_texts = [text for text in texts if len(text["segments"]) > 1]
    non_segmented_texts = [text for text in texts if len(text["segments"]) == 1]

    # Calculate number of samples for each category
    n_segmented = int(n_samples * segmented_ratio)
    n_non_segmented = n_samples - n_segmented

    # Adjust if we don't have enough texts in either category
    n_segmented = min(n_segmented, len(segmented_texts))
    n_non_segmented = min(n_non_segmented, len(non_segmented_texts))

    # Sample texts from each category
    sampled_segmented = random.sample(segmented_texts, n_segmented)
    sampled_non_segmented = random.sample(non_segmented_texts, n_non_segmented)

    # Combine samples
    sampled_texts = sampled_segmented + sampled_non_segmented
    random.shuffle(sampled_texts)  # Shuffle to mix the two categories

    # Get list of segment counts for statistics
    segment_counts_list = [len(text["segments"]) for text in texts]
    segment_counts = Counter(segment_counts_list)

    # Calculate statistics
    mean_segments = statistics.mean(segment_counts_list)
    median_segments = statistics.median(segment_counts_list)
    mode_segments = statistics.mode(segment_counts_list)
    try:
        stdev_segments = statistics.stdev(segment_counts_list)
    except:
        stdev_segments = 0

    # Print detailed statistics for full dataset
    print("\nFull dataset statistics:")
    print(f"Total texts: {len(texts)}")
    print(f"Segmented texts (>1 segment): {len(segmented_texts)}")
    print(f"Non-segmented texts (1 segment): {len(non_segmented_texts)}")
    print(f"Mean segments per text: {mean_segments:.2f}")
    print(f"Median segments per text: {median_segments}")
    print(f"Mode segments per text: {mode_segments}")
    print(f"Standard deviation: {stdev_segments:.2f}")
    print(f"Min segments: {min(segment_counts_list)}")
    print(f"Max segments: {max(segment_counts_list)}")

    print("\nDetailed segment count distribution:")
    total_texts = len(texts)
    for n_segments, count in sorted(segment_counts.items()):
        percentage = (count / total_texts) * 100
        print(f"Texts with {n_segments} segments: {count} ({percentage:.1f}%)")

    # Get stats for sampled texts
    sampled_counts_list = [len(text["segments"]) for text in sampled_texts]
    sampled_segment_counts = Counter(sampled_counts_list)

    # Calculate statistics for sample
    sample_mean = statistics.mean(sampled_counts_list)
    sample_median = statistics.median(sampled_counts_list)
    sample_mode = statistics.mode(sampled_counts_list)
    try:
        sample_stdev = statistics.stdev(sampled_counts_list)
    except:
        sample_stdev = 0

    print("\nSampled texts statistics:")
    print(f"Total sampled texts: {len(sampled_texts)}")
    print(f"Sampled segmented texts (>1 segment): {n_segmented}")
    print(f"Sampled non-segmented texts (1 segment): {n_non_segmented}")
    print(f"Mean segments per text: {sample_mean:.2f}")
    print(f"Median segments per text: {sample_median}")
    print(f"Mode segments per text: {sample_mode}")
    print(f"Standard deviation: {sample_stdev:.2f}")
    print(f"Min segments: {min(sampled_counts_list)}")
    print(f"Max segments: {max(sampled_counts_list)}")

    print("\nSampled texts segment count distribution:")
    for n_segments, count in sorted(sampled_segment_counts.items()):
        percentage = (count / len(sampled_texts)) * 100
        print(f"Texts with {n_segments} segments: {count} ({percentage:.1f}%)")

    # Write sampled texts
    with open(output_file, "w", encoding="utf-8") as f:
        for text in sampled_texts:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 10_sample_for_evaluation.py <input_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = (
        sys.argv[2]
        if len(sys.argv) == 3
        else input_path.replace(".jsonl", "_sample.jsonl")
    )

    sample_mixed_texts(
        input_path,
        output_path,
        n_samples=100,
        segmented_ratio=0.8,
    )


if __name__ == "__main__":
    main()
