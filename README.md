# Fuzzy Sarcasm Classifier

This project implements a system to classify **how sarcastic** a given Twitter comment is using a fuzzy approach. It consists of two parts:

1. A binary classifier that predicts the probability that a comment is sarcastic.
2. A fuzzy layer that maps that probability to three overlapping fuzzy sets using trapezoidal membership functions:
   - **Unlikely**: `fuzz.trapmf(x, [0.0, 0.0, 0.3, 0.5])`
   - **Probably**: `fuzz.trapmf(x, [0.3, 0.5, 0.6, 0.8])`
   - **Highly Likely**: `fuzz.trapmf(x, [0.6, 0.8, 1.0, 1.0])`

The output for a given comment includes:
- The sarcasm probability (a value in [0, 1])
- The fuzzy membership degrees for the three categories

## File Structure
fuzzy_sarcasm_classifier/ 
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv      # Training data (columns: comment, label)
│   └── test.csv       # Test data (columns: comment, label)
├── notebooks/
│   └── demo.ipynb     # Notebook demonstrating training, fuzzy classification, and visualization
└── src/
    ├── __init__.py
    ├── data_loader.py         # Functions to load CSV data
    ├── sarcasm_classifier.py  # Sarcasm classifier using fuzzy trapezoidal membership functions
    └── utils.py               # Utility functions (e.g., text cleaning)

## Setup and Running

1. **Install dependencies:**  
   From the project root run:
   ```bash
   pip install -r requirements.txt