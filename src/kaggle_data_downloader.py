import shutil
import kagglehub
import os

# Content
# The dataset consists of 2 files, train.csv and test.csv.
#
# Both these files contain 2 columns:
#
# comment: The text of the tweet
# label: The respective class to which the tweet belongs. There are 4 classes -:
#
# Irony
# Sarcasm
# Regular
# Figurative (both irony and sarcasm)
#
# Reference
# Jennifer Ling, Roman Klinger: “An Empirical, Quantitative Analysis of the Differences between Sarcasm and Irony”.
# Semantic Sentiment and Emotion Workshop, ESWC, Crete. Greece. 2016

# Download latest version
path = kagglehub.dataset_download("nikhiljohnk/tweets-with-sarcasm-and-irony")

print("Path to dataset files:", path)


# Define the destination directory
destination_dir = "data"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Move the downloaded files to the destination directory
for file_name in os.listdir(path):
    full_file_name = os.path.join(path, file_name)
    if os.path.isfile(full_file_name):
        shutil.move(full_file_name, destination_dir)
