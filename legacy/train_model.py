import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model(train_csv: str, val_csv: str, max_iter: int = 200):
    """
    Trains an MLP on the IK dataset and evaluates on validation set.
    Saves the trained model and scaler to disk.
    """
    # Load data
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)

    X_train = df_train[['x','y','z','psi']].values
    y_train = df_train[['t1','t2','t3','t4']].values
    X_val   = df_val[['x','y','z','psi']].values
    y_val   = df_val[['t1','t2','t3','t4']].values

    # Scale inputs
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # Define and train model
    model = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation MSE: {mse:.6f}")

    # Save artifacts
    joblib.dump(model, 'ik_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved model to 'ik_model.pkl' and scaler to 'scaler.pkl'")

# command line 
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='Train IK MLP model.')
#     parser.add_argument('--train', type=str,   default='train.csv')
#     parser.add_argument('--val',   type=str,   default='val.csv')
#     parser.add_argument('--max_iter', type=int, default=200)
#     args = parser.parse_args()
#     train_model(args.train, args.val, args.max_iter)
