# Macbook Pro M4 Max Performance

In terms of ML / DL / Language model, performance measurement in comparison to Nvidia RTX 4090.

## Softmax Calculation
Logit Matrix N = 60,000
W Matrix N = 60,000

### Mac
About 50 seconds
Using Torch MPS (Apple GPU) implementation

### NVIDIA RTX 4090
About 1.8 seconds
Using Torch GPU (RTX 4090) implementatoin


## Sigmoid Calculation

Logit Matrix N = 60,000
W Matrix N = 60,000

### Mac
About 48 seconds
Using Torch MPS (Apple GPU) implementation

### NVIDIA RTX 4090
About 2.03 seconds
Using Torch GPU (RTX 4090) implementatoin
