import numpy as np
import pickle
import skfuzzy as fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class SarcasmClassifier:
    def __init__(self):
        # Initialize vectorizer and binary classifier
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LogisticRegression(solver='liblinear')

        # Define an x-range for sarcasm probability (from 0 to 1)
        self.x_sarcasm = np.linspace(0, 1, 100)

        # Define trapezoidal membership functions for sarcasm likelihood
        self.mf_unlikely = fuzz.trapmf(self.x_sarcasm, [0.0, 0.0, 0.3, 0.5])
        self.mf_probably = fuzz.trapmf(self.x_sarcasm, [0.3, 0.5, 0.6, 0.8])
        self.mf_highly_likely = fuzz.trapmf(self.x_sarcasm, [0.6, 0.8, 1.0, 1.0])

    def train(self, texts, labels):
        """
        Train the sarcasm classifier.
        A comment is considered sarcastic if its label is 'sarcasm' or 'figurative' (both sarcasm and irony).
        """
        # Convert labels to binary: 1 if sarcasm, else 0
        y = np.array([1 if label.lower() == 'sarcasm' or label.lower() == 'figurative' else 0 for label in labels])
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)

    def predict_proba(self, texts):
        """
        Predict the probability of sarcasm for each text.
        Returns a NumPy array of probabilities (values in [0, 1]).
        """
        X = self.vectorizer.transform(texts)
        # Assumes classifier.classes_ == [0, 1]
        proba = self.classifier.predict_proba(X)
        # Return probability for class 1 (sarcasm)
        return proba[:, 1]

    def classify_fuzzy(self, texts):
        """
        For each text, predict the sarcasm probability and compute fuzzy membership degrees
        for:
         - 'Unlikely'
         - 'Probably'
         - 'Highly Likely'

        Returns a list of dictionaries for each text.
        """
        probabilities = self.predict_proba(texts)
        results = []
        for prob in probabilities:
            # Compute membership degrees using interpolation
            degree_unlikely = fuzz.interp_membership(self.x_sarcasm, self.mf_unlikely, prob)
            degree_probably = fuzz.interp_membership(self.x_sarcasm, self.mf_probably, prob)
            degree_highly_likely = fuzz.interp_membership(self.x_sarcasm, self.mf_highly_likely, prob)
            results.append({
                'sarcasm_probability': prob,
                'unlikely': degree_unlikely,
                'probably': degree_probably,
                'highly_likely': degree_highly_likely
            })
        return results

    def save_model(self, path):
        """
        Save the vectorizer and classifier to a pickle file.
        """
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'classifier': self.classifier}, f)

    def load_model(self, path):
        """
        Load the vectorizer and classifier from a pickle file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']