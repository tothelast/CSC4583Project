# This repository is designed for the final project of the CSC 483 and CSC 583 students

## Contributors:

-   **Garegin Mazmanyan**
-   **Chitrangada Juneja**
-   **Adan Baca**
-   **Simi Saha**

## Dataset used:

-   https://huggingface.co/datasets/orionweller/NevIR

## Paper to read:

-   https://arxiv.org/pdf/2305.07614

## Meeting Recordings:

-   Meeting 1: https://drive.google.com/file/d/1JrKrUQLvV0LlCn-yzw8llw6wqav_DI_e/view?usp=sharing

# NevIR Project

This project implements models for handling negation in information retrieval tasks.

## Models

### MonoT5 Model

The MonoT5 model is a sequence-to-sequence transformer that excels at understanding negation in search queries. Unlike traditional retrieval models, MonoT5 directly scores document-query pairs by predicting "true" or "false" to indicate relevance. It uses a cross-encoder architecture, which allows it to capture complex relationships between negated queries and documents.

Our implementation adapts the state-of-the-art MonoT5-3B model from the Pygaggle library, optimized for Apple Silicon hardware. This model achieved the highest performance (~50% pairwise accuracy) in the original NevIR paper by effectively understanding how negation changes the semantic meaning of queries. When run on M3 Mac hardware, our implementation achieved 51% pairwise accuracy.

When a user adds negation to a query (e.g., changing "previously inhabited islands" to "previously uninhabited islands"), MonoT5 correctly flips the document rankings to match the semantic meaning shift.
