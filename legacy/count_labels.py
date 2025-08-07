# sanity check to analyze the counts of each label in classification_dataset.csv

import pandas as pd
from config_classifier import DATA_PATH

def main():
    # Load the classification dataset
    df = pd.read_csv(DATA_PATH)

    # Count occurrences of each label
    counts = df['label'].value_counts().sort_index()

    print("Label counts:")
    for label, count in counts.items():
        print(f"  Class {label}: {count}")
    print(f"Total samples: {len(df)}")

if __name__ == "__main__":
    main()
