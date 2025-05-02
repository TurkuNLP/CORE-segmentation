import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any


def load_data(filepath: str) -> Dict[str, Any]:
    """Load and return the processed features."""
    with open(filepath, "r") as f:
        return json.load(f)


def prepare_data(
    data: List[Dict[str, Any]], feature_list: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Convert data to feature matrix and list of registers."""
    # Extract features and registers
    X = np.array([list(item["features"].values()) for item in data])
    registers = [item["register"] for item in data]
    return X, registers


def compute_within_register_variance(
    X: np.ndarray, registers: List[str]
) -> Dict[str, float]:
    """Compute the average variance of features within each register."""
    unique_registers = list(set(registers))
    variances = {}

    for reg in unique_registers:
        # Get data for this register
        mask = np.array(registers) == reg
        if sum(mask) > 1:  # Need at least 2 samples
            reg_data = X[mask]
            # Compute variance for each feature and take mean
            variances[reg] = np.mean(np.var(reg_data, axis=0))

    return variances


def compute_average_pairwise_distances(
    X: np.ndarray, registers: List[str]
) -> Dict[str, float]:
    """Compute average pairwise distances within each register."""
    unique_registers = list(set(registers))
    distances = {}

    for reg in unique_registers:
        # Get data for this register
        mask = np.array(registers) == reg
        if sum(mask) > 1:  # Need at least 2 samples
            reg_data = X[mask]
            # Compute pairwise distances
            dist_matrix = squareform(pdist(reg_data))
            # Get average of upper triangle
            distances[reg] = np.mean(
                dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            )

    return distances


def analyze_cohesion(data_path: str = "processed_features.json"):
    """Analyze register cohesion in segments vs full texts."""
    # Load data
    data = load_data(data_path)
    feature_list = data["features"]

    # Prepare data for segments and full texts
    X_segments, registers_segments = prepare_data(data["segments"], feature_list)
    X_full, registers_full = prepare_data(data["full_texts"], feature_list)

    # Standardize features
    scaler = StandardScaler()
    X_segments_scaled = scaler.fit_transform(X_segments)
    X_full_scaled = scaler.transform(X_full)

    # Compute metrics for both segments and full texts
    results = {
        "segments": {
            "variance": compute_within_register_variance(
                X_segments_scaled, registers_segments
            ),
            "distances": compute_average_pairwise_distances(
                X_segments_scaled, registers_segments
            ),
            "silhouette": (
                silhouette_score(X_segments_scaled, registers_segments)
                if len(set(registers_segments)) > 1
                else 0
            ),
        },
        "full_texts": {
            "variance": compute_within_register_variance(X_full_scaled, registers_full),
            "distances": compute_average_pairwise_distances(
                X_full_scaled, registers_full
            ),
            "silhouette": (
                silhouette_score(X_full_scaled, registers_full)
                if len(set(registers_full)) > 1
                else 0
            ),
        },
    }

    # UMAP for visualization
    reducer = umap.UMAP(random_state=42)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot segments
    embedding_segments = reducer.fit_transform(X_segments_scaled)
    scatter1 = ax1.scatter(
        embedding_segments[:, 0],
        embedding_segments[:, 1],
        c=[REGISTER_COLORS.get(r, "#000000") for r in registers_segments],
        alpha=0.6,
    )
    ax1.set_title("Segments")

    # Plot full texts
    embedding_full = reducer.fit_transform(X_full_scaled)
    scatter2 = ax2.scatter(
        embedding_full[:, 0],
        embedding_full[:, 1],
        c=[REGISTER_COLORS.get(r, "#000000") for r in registers_full],
        alpha=0.6,
    )
    ax2.set_title("Full Texts")

    # Add legend
    unique_registers = list(set(registers_segments + registers_full))
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=REGISTER_COLORS.get(reg, "#000000"),
            label=reg,
            markersize=10,
        )
        for reg in unique_registers
    ]
    fig.legend(handles=legend_elements, loc="center right")

    plt.tight_layout()
    plt.savefig("register_clusters.png", dpi=300, bbox_inches="tight")

    # Print results
    print("\nResults Summary:")
    print("\nSilhouette Scores (higher is better):")
    print(f"Segments: {results['segments']['silhouette']:.3f}")
    print(f"Full Texts: {results['full_texts']['silhouette']:.3f}")

    print("\nAverage Within-Register Variances (lower is better):")
    for reg in unique_registers:
        seg_var = results["segments"]["variance"].get(reg, float("nan"))
        full_var = results["full_texts"]["variance"].get(reg, float("nan"))
        print(f"{reg:>3}: Segments = {seg_var:.3f}, Full Texts = {full_var:.3f}")

    print("\nAverage Pairwise Distances (lower is better):")
    for reg in unique_registers:
        seg_dist = results["segments"]["distances"].get(reg, float("nan"))
        full_dist = results["full_texts"]["distances"].get(reg, float("nan"))
        print(f"{reg:>3}: Segments = {seg_dist:.3f}, Full Texts = {full_dist:.3f}")

    return results


# Color mapping for registers
REGISTER_COLORS = {
    "LY": "#1f77b4",  # blue
    "SP": "#ff7f0e",  # orange
    "ID": "#2ca02c",  # green
    "NA": "#d62728",  # red
    "HI": "#9467bd",  # purple
    "IN": "#8c564b",  # brown
    "OP": "#e377c2",  # pink
    "IP": "#7f7f7f",  # gray
}

if __name__ == "__main__":
    results = analyze_cohesion()
