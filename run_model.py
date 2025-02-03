#!/usr/bin/env python
import argparse
import sys
from src.sarcasm_classifier import SarcasmClassifier


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Predict the level of sarcasm in a comment.")
    parser.add_argument("comment", type=str, help="The comment text to classify.")
    parser.add_argument("--model", type=str, default="sarcasm_model.pkl",
                        help="Path to the saved model pickle file (default: sarcasm_model.pkl)")
    args = parser.parse_args()

    # Initialize the classifier and load the saved model
    sarcasm_clf = SarcasmClassifier()
    try:
        sarcasm_clf.load_model(args.model)
    except Exception as e:
        sys.exit(f"Error loading model from {args.model}: {e}")

    # Predict the sarcasm using fuzzy classification
    result = sarcasm_clf.classify_fuzzy([args.comment])[0]

    # Print the result
    print(f"Input Comment: {args.comment}")
    print(f"Sarcasm Probability: {result['sarcasm_probability']:.2f}")
    print("Fuzzy Classification:")
    print(f"  Unlikely:      {result['unlikely']:.2f}")
    print(f"  Probably:      {result['probably']:.2f}")
    print(f"  Highly Likely: {result['highly_likely']:.2f}")


if __name__ == "__main__":
    main()