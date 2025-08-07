import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(input_csv: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Splits a raw CSV into train/val/test based on provided ratios.
    """
    df = pd.read_csv(input_csv)
    test_ratio = 1.0 - train_ratio - val_ratio
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    df_train, df_tmp = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=42)
    relative_val = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(df_tmp, test_size=(1.0 - relative_val), random_state=42)
    return df_train, df_val, df_test

# Command line
def main():
    parser = argparse.ArgumentParser(description='Split raw dataset into train/val/test.')
    parser.add_argument('input_csv', type=str, help='Raw dataset CSV file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Fraction for training set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Fraction for validation set')
    args = parser.parse_args()

    df_train, df_val, df_test = split_dataset(args.input_csv, args.train_ratio, args.val_ratio)
    df_train.to_csv('train.csv', index=False)
    df_val.to_csv('val.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    print('Saved train.csv, val.csv, test.csv')

if __name__ == '__main__':
    main()
