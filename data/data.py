from datasets import load_dataset

def load_nevir_dataset():
    """
    Load the NevIR dataset from Hugging Face.

    Returns:
        tuple: (train_set, dev_set, test_set) containing the three splits of the dataset
    """
    train_set = load_dataset("orionweller/nevir", split="train")
    val_set = load_dataset("orionweller/nevir", split="validation")
    test_set = load_dataset("orionweller/nevir", split="test")

    return train_set, val_set, test_set




