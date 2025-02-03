#!/usr/bin/env python
import os
import argparse
import pandas as pd
from src.data_loader import load_data
from src.sarcasm_classifier import SarcasmClassifier
from src.utils import preprocess_texts

def main():
    parser = argparse.ArgumentParser(
        description="Train the sarcasm classifier and save the model to a pickle file."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "train.csv"),
        help="Path to the training CSV file (default: data/train.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sarcasm_model.pkl",
        help="Path to save the trained model (default: sarcasm_model.pkl)",
    )
    args = parser.parse_args()

    # Load training data
    try:
        df = load_data(args.data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Check that the required columns exist
    if "comment" not in df.columns or "label" not in df.columns:
        print("Error: Training data must have 'comment' and 'label' columns.")
        return

    # Preprocess the comments
    df["clean_comment"] = preprocess_texts(df["comment"])

    # Initialize and train the sarcasm classifier
    sarcasm_clf = SarcasmClassifier()
    sarcasm_clf.train(df["clean_comment"], df["label"])
    print("Sarcasm classifier trained successfully.")

    # Save the model using pickle
    sarcasm_clf.save_model(args.model)
    print(f"Model saved to {args.model}")

if __name__ == "__main__":
    main()