import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data import load_nevir_dataset
from models import RandomModel, MonoT5Model
from tf_idf_model import tfidfModel
from negation_model import negationModel
from evaluate import calculate_pairwise_accuracy


_,_, test_set = load_nevir_dataset()


random = RandomModel()
tfidf = tfidfModel([])
negation = negationModel()
monot5 = MonoT5Model(model_name="castorini/monot5-3b-msmarco-10k")

accuracies = {}  

print("calculating acc for random")
accuracies["Random"] = calculate_pairwise_accuracy(random, test_set)
print("calculating acc for tfidf")
accuracies["TF-IDF"] = calculate_pairwise_accuracy(tfidf, test_set)
print("calculating acc for negation")
accuracies["Negation Heuristic"] = calculate_pairwise_accuracy(negation, test_set)
print("calculating acc for mono")
accuracies["MonoT5"] = calculate_pairwise_accuracy(monot5, test_set)

names = list(accuracies.keys())
scores = list(accuracies.values())
cmap = plt.get_cmap('tab10')
colors = cmap(np.arange(len(names)))

plt.figure(figsize=(8,5))
bars = plt.bar(names, scores, color=colors)
plt.ylim(0,1)
plt.ylabel("Pairwise Accuracy")
plt.title("Model Pairwise Accuracy on NevIR Test Set")
plt.xticks(rotation=45, ha="right")

for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, score+0.02, f"{score:.2%}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png')
plt.show()




