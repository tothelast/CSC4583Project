"""
@author: Garegin Mazmanyan
Evaluation script for NevIR models that calculates pairwise accuracy.

As described in the paper, pairwise accuracy checks if a model correctly ranks
both documents in a pair for both queries (flipping rankings appropriately).
"""
import sys
import os
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data import load_nevir_dataset
from models import RandomModel, MonoT5Model
from tf_idf_model import tfidfModel
from negation_model import negationModel

def calculate_pairwise_accuracy(model, dataset):
    """
    Calculate the pairwise accuracy metric described in the NevIR paper.

    Args:
        model: A model with a rank_documents method
        dataset: The NevIR dataset with paired queries and documents

    Returns:
        float: The pairwise accuracy for the dataset
    """
    correct_pairs = 0
    total_pairs = 0

    for example in dataset:
        query1 = example['q1']
        query2 = example['q2']
        documents = [example['doc1'], example['doc2']]

        # Get rankings for both queries
        ranking1 = model.rank_documents(query1, documents)
        ranking2 = model.rank_documents(query2, documents)

        # Extract just the indices from the rankings (MonoT5 returns (idx, score) tuples)
        if isinstance(ranking1[0], tuple):
            indices1 = [idx for idx, _ in ranking1]
            indices2 = [idx for idx, _ in ranking2]
        else:
            indices1 = ranking1
            indices2 = ranking2

        # Check if rankings are correctly flipped
        if indices1[0] == 0 and indices2[0] == 1:
            correct_pairs += 1

        total_pairs += 1

    # Return accuracy as a percentage
    return correct_pairs / total_pairs


def main():
    # Load the NevIR dataset
    _, _, test_set = load_nevir_dataset()

    # Initialize models
    # random_model = RandomModel()
    monot5_model = MonoT5Model(model_name="castorini/monot5-3b-msmarco-10k")

    # Calculate and print accuracy for MonoT5 (the best model from the paper)
    accuracy_monot5 = calculate_pairwise_accuracy(monot5_model, test_set)
    print(f"MonoT5 model pairwise accuracy: {accuracy_monot5:.2f}")

    # Uncomment these lines to test other models
    # tfidf_model = tfidfModel([])
    # accuracy_tfidf = calculate_pairwise_accuracy(tfidf_model, test_set)
    # print(f"TF-IDF model pairwise accuracy: {accuracy_tfidf:.2f}")

    # negation_model = negationModel()
    # accuracy_negation = calculate_pairwise_accuracy(negation_model, test_set)
    # print(f"Negation model pairwise accuracy: {accuracy_negation:.2f}")

    # accuracy_random = calculate_pairwise_accuracy(random_model, test_set)
    # print(f"Random model pairwise accuracy: {accuracy_random:.2f}")


if __name__ == "__main__":
    main()
