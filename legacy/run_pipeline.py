import argparse
from preprocess import split_dataset
from train_model import train_model

def main():
    parser = argparse.ArgumentParser(
        description="Split raw data, train MLP, and save artifacts."
    )
    parser.add_argument("input_csv", help="Raw dataset CSV")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--max_iter",    type=int,   default=200)
    args = parser.parse_args()

    # split
    df_train, df_val, df_test = split_dataset(
        args.input_csv, args.train_ratio, args.val_ratio
    )

    # save the splits (optional)
    df_train.to_csv("train.csv", index=False)
    df_val.to_csv("val.csv",     index=False)
    df_test.to_csv("test.csv",   index=False)

    # train
    train_model("train.csv", "val.csv", args.max_iter)

# python run_pipeline.py raw_dataset.csv --train_ratio 0.7 --val_ratio 0.2 --max_iter 300
if __name__ == "__main__":
    main()
