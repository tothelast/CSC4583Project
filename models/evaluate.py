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
from models import RandomModel

def calculate_pairwise_accuracy(model, dataset):
    """
    Calculate the pairwise accuracy metric described in the NevIR paper.

    Args:
        model: A model with a rank_documents method
        dataset: The NevIR dataset to evaluate on

    Returns:
        float: The pairwise accuracy score (0-1)
    """
    correct_pairs = 0

    for example in dataset:
        query1 = example['q1']
        query2 = example['q2']
        documents = [example['doc1'], example['doc2']]

        # Get rankings for both queries
        ranking1 = model.rank_documents(query1, documents)
        ranking2 = model.rank_documents(query2, documents)

        # Check if the rankings are correct
        if ranking1[0] == 0 and ranking2[0] == 1:
            correct_pairs += 1

    # Calculate accuracy
    return correct_pairs / len(dataset)

def main():
    # Load the test set
    _, _, test_set = load_nevir_dataset()

    # Initialize the random model
    model = RandomModel()

    # Calculate pairwise accuracy
    accuracy = calculate_pairwise_accuracy(model, test_set)

    print(f"Random model pairwise accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
