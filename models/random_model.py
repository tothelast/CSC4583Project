"""
Random baseline model for NevIR.

This model randomly ranks two documents, which should give 25% pairwise accuracy
as mentioned in the NevIR paper (two pairs with random ranking).
"""
import random

class RandomModel:
    """
    A random baseline model for NevIR that randomly ranks documents.

    According to the paper, random ranking should achieve 25% pairwise accuracy
    since there are two document pairs and random chance of correct ordering is 0.5*0.5 = 0.25.
    """

    def __init__(self):
        """Initialize the random model."""
        pass

    def rank_documents(self, query, documents):
        """
        Randomly rank the documents for a given query.

        Args:
            query (str): The query text
            documents (list): List of document texts

        Returns:
            list: A list of document indices sorted by rank (highest first)
        """
        # Create a list of document indices
        doc_indices = list(range(len(documents)))

        # Shuffle the indices randomly
        random.shuffle(doc_indices)

        return doc_indices
