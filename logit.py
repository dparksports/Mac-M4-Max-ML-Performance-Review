import torch
import numpy as np
import time
from tqdm import tqdm

count = 60000 
N = count
D = count 

# Generate random data
np.random.seed(0)
X = np.random.rand(N, D).astype(np.float64)
y = np.random.randint(2, size=N).astype(np.float64)

# Logit-based binary classification
start_time = time.time()
logits = np.dot(X, np.random.rand(D).astype(np.float64))
logit_probabilities = 1 / (1 + np.exp(-logits))
logit_accuracy = np.mean((logit_probabilities > 0.5) == y)
logit_time = time.time() - start_time


# Progress tick
pbar = tqdm(total=3, desc="Calculations")
pbar.set_description("Logit-based")
pbar.update(1)
pbar.close()

print("Logit-based Accuracy:")
print(f"{logit_accuracy:.5f}")
print(f"Time: {logit_time:.2f} seconds")
print()


print("Performance Comparison:")
print(f"Logit-based: {logit_time:.2f} seconds")

