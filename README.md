# AI/ML Intern Mini-Assessment

## Contents

1. `iris_classifier.py` - Two-layer neural network using PyTorch on the Iris dataset.
2. `generate.py` - Text generation script using GPT-2 from Hugging Face.
3. `accuracy_vs_epoch.png` - Plot of accuracy over training epochs.
4. `samples.txt` - Output samples generated at temperature 0.7 and 1.0.
5. `mini_report.md` - Summary of architecture, results, and key learnings.

## How to Run

### Part 1: Classification
```bash
pip install torch scikit-learn matplotlib
python iris_classifier.py
```

### Part 2: Generative AI
```bash
pip install transformers torch
python generate.py
```

All scripts are self-contained and run from command line. Outputs are saved in the same directory.
