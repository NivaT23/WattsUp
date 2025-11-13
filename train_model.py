# train_model.py
import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def find_column(df, keywords):
    """Return first column name containing any keyword (case-insensitive)."""
    for col in df.columns:
        low = col.lower()
        for kw in keywords:
            if kw in low:
                return col
    return None

def to_numeric(series):
    """Convert a pandas Series to numeric, removing commas, rupee symbols, stray text."""
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce')

def main(filename="bill_data.csv"):
    print(f"Loading {filename} ...")
    df = pd.read_csv(filename)
    print("Columns found:", list(df.columns))

    # Try to detect relevant columns
    units_col = find_column(df, ["unit", "units", "consumption", "kwh"])
    bill_col = find_column(df, ["bill", "amount", "price", "cost", "rupee", "₹"])

    if units_col is None or bill_col is None:
        raise SystemExit("Could not automatically find 'units' or 'bill' columns. Please check the CSV headers.")

    print("Using columns:", units_col, "->", bill_col)

    # Clean
    df[units_col] = to_numeric(df[units_col])
    df[bill_col] = to_numeric(df[bill_col])

    df = df[[units_col, bill_col]].dropna()
    df.columns = ["Units", "Bill"]  # rename for consistency

    if len(df) < 5:
        print("Warning: small dataset (less than 5 rows). Regression may be unstable, but will still run.")

    # Train/test split to show simple evaluation (optional)
    X = df[["Units"]].values
    y = df["Bill"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate quick
    preds = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, preds)) if len(y_test)>0 else float("nan")
    print(f"Trained LinearRegression. Test RMSE: {rmse:.2f}")

    # Save model
    joblib.dump(model, "bill_model.pkl")
    print("Saved trained model to bill_model.pkl")

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "bill_data.csv"
    main(filename)
