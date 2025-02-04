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
```bash
{project_root}/ 
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
    ├── kagggle_data_loader.py # Script that loads dataset from Kaggle (which is already included in this project)
    ├── sarcasm_classifier.py  # Sarcasm classifier using fuzzy trapezoidal membership functions
    └── utils.py               # Utility functions for text cleaniing before processing
```
## Setup and Running

1. **Install dependencies:**  
   From the project root run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the classifier:**  
   From the project root run:
   ```bash
   python -m src.sarcasm_classifier
   ```
   This will train the classifier using the training data and save the model to `src/sarcasm_classifier.pkl`.
3.  **Run the demo notebook:**
    ```bash
    jupyter notebook notebooks/demo.ipynb
    ```
4. For testing/optionally run the model with your own message from the command line:
   ```bash
   python run_model.py "What a great day!" 
   
   Input Comment: What a great day
   Sarcasm Probability: 0.58
   Fuzzy Classification:
   Unlikely:      0.00
   Probably:      1.00
   Highly Likely: 0.00

   ```
## Documentation


