import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import CrossEncoder
from sklearn.metrics import average_precision_score
from typing import List

# Define evaluation metrics
def mean_reciprocal_rank(relevance_labels: List[int], scores: List[float]) -> float:
    sorted_indices = np.argsort(scores)[::-1]
    for rank, idx in enumerate(sorted_indices, start=1):
        if relevance_labels[idx] == 1:
            return 1 / rank
    return 0

def mean_average_precision(relevance_labels: List[int], scores: List[float]) -> float:
    return average_precision_score(relevance_labels, scores)

def ndcg_at_k(relevance_labels: List[int], scores: List[float], k: int = 10) -> float:
    sorted_indices = np.argsort(scores)[::-1]
    relevance_sorted = np.take(relevance_labels, sorted_indices[:k])
    dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevance_sorted))
    idcg = sum(1 / np.log2(rank + 2) for rank in range(min(k, sum(relevance_labels))))
    return dcg / idcg if idcg > 0 else 0

# Load the datasets and define the models
datasets = {
    "Relevance_Labels_Dataset": load_dataset("NAMAA-Space/Ar-Reranking-Eval")["train"],
    "Positive_Negatives_Dataset": load_dataset("NAMAA-Space/Arabic-Reranking-Triplet-5-Eval")["train"]
}

models = {
    "CrossEncoder-arabic-GATE": CrossEncoder("NAMAA-Space/GATE-Reranker-V1")
}

# Evaluation for each model and dataset
results = []
sample_predictions = []

for dataset_name, dataset in datasets.items():
    print(f"Evaluating on dataset: {dataset_name}")
    sample_queries = random.sample(dataset.to_pandas()['query'].unique().tolist(), 5)  # Sample 5 queries for inspection

    for model_name, model in models.items():
        print(f"Evaluating model: {model_name} on {dataset_name}")
        all_mrr, all_map, all_ndcg = [], [], []

        # Determine dataset structure (Dataset 1: relevance labels or Dataset 2: positive/negatives)
        if 'candidate_document' in dataset.column_names:
            grouped_data = dataset.to_pandas().groupby("query")
            for query, group in grouped_data:
                candidate_texts = group['candidate_document'].tolist()
                relevance_labels = group['relevance_label'].tolist()

                # Get scores from model
                pairs = [(query, doc) for doc in candidate_texts]
                scores = model.predict(pairs)

                # Calculate metrics
                all_mrr.append(mean_reciprocal_rank(relevance_labels, scores))
                all_map.append(mean_average_precision(relevance_labels, scores))
                all_ndcg.append(ndcg_at_k(relevance_labels, scores, k=10))

                # Collect sample predictions
                if query in sample_queries:
                    for doc, score, label in zip(candidate_texts, scores, relevance_labels):
                        sample_predictions.append({
                            "dataset": dataset_name,
                            "model": model_name,
                            "query": query,
                            "candidate_document": doc,
                            "predicted_score": score,
                            "relevance_label": label
                        })

        else:  # For Positive-Negative Structure
            for entry in dataset:
                query = entry['query']
                candidate_texts = [entry['positive'], entry['negative1'], entry['negative2'], entry['negative3'], entry['negative4']]
                relevance_labels = [1, 0, 0, 0, 0]

                # Get scores from model
                pairs = [(query, doc) for doc in candidate_texts]
                scores = model.predict(pairs)

                # Calculate metrics
                all_mrr.append(mean_reciprocal_rank(relevance_labels, scores))
                all_map.append(mean_average_precision(relevance_labels, scores))
                all_ndcg.append(ndcg_at_k(relevance_labels, scores, k=10))

                # Collect sample predictions
                if query in sample_queries:
                    for doc, score, label in zip(candidate_texts, scores, relevance_labels):
                        sample_predictions.append({
                            "dataset": dataset_name,
                            "model": model_name,
                            "query": query,
                            "candidate_document": doc,
                            "predicted_score": score,
                            "relevance_label": label
                        })

        # Aggregate results for the model on this dataset
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "MRR": np.mean(all_mrr),
            "MAP": np.mean(all_map),
            "nDCG@10": np.mean(all_ndcg)
        })

# Save aggregated results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("combined_model_evaluation_results.csv", index=False)

# Save sample predictions to CSV for inspection
sample_predictions_df = pd.DataFrame(sample_predictions)
sample_predictions_df.to_csv("combined_sample_predictions.csv", index=False)

print("Evaluation completed. Results saved to combined_model_evaluation_results.csv and combined_sample_predictions.csv.")