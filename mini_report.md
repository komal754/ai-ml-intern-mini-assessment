# Mini Report: AI/ML Intern Assessment

## ML Classification

**Model**: Two-layer neural network  
**Hyperparameters**:  
- Learning rate: 0.01  
- Epochs: 50  
- Activation: ReLU  
- Optimizer: SGD

**Results**:
- Train Accuracy: ~98%
- Test Accuracy: ~96.7%

### Interpretation
The model generalizes well with minimal overfitting. The manual training loop helped deepen understanding of backpropagation.

## Generative AI Experiment

**Prompt**: "Once upon a time in a land far away,"  
**Model**: GPT-2  
**Sampling**: Top-k (k=50)

### Temperature Observations
- **0.7**: More structured and predictable story generation.
- **1.0**: Increased randomness and creativity, but sometimes less coherent.

## Key Learnings
- Implementing neural nets from scratch gives deep insights into model internals.
- GPT-2's outputs vary significantly with temperature — it's an intuitive control for creativity.
