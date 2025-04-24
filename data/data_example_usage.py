"""
@author: Garegin Mazmanyan
Example script showing how to import and use the NevIR dataset in other files.
"""
from data import load_nevir_dataset

def main():
    # Load all dataset splits
    train_set, val_set, test_set = load_nevir_dataset()
    print(f"Loaded dataset with {len(train_set)} training examples, {len(val_set)} validation examples, and {len(test_set)} test examples.")

if __name__ == "__main__":
    main()
