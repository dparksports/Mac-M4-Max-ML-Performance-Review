import torch
import numpy as np
import time
from tqdm import tqdm

count = 80000 
N = count
D = count 

# Generate random data
np.random.seed(0)
X = np.random.rand(N, D).astype(np.float32)  # Convert to float32 for PyTorch
y = np.random.randint(2, size=N).astype(np.float32)


# PyTorch softmax-based binary classification
start_time = time.time()
X_tensor = torch.from_numpy(X)
weights = torch.randn(D, 2, requires_grad=True)  # Initialize weights for softmax
logits = torch.matmul(X_tensor, weights)
softmax_probabilities = torch.softmax(logits, dim=1)
# Convert probabilities to binary predictions (0 or 1)
softmax_predictions = torch.argmax(softmax_probabilities, dim=1)
softmax_time = time.time() - start_time

# Progress tick
pbar = tqdm(total=1, desc="Calculations")
pbar.set_description("Softmax-based")
pbar.update(1)
pbar.close()

print("Performance Comparison:")
print(f"Softmax-based: {softmax_time:.2f} seconds")