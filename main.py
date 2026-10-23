import os
from src.data_loader import load_data
from src.preprocessing import split_and_scale
from src.model import train_model
from src.evaluate import evaluate_model, save_predictions
from src.visualize import plot_regression_line, plot_residuals


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data
    df = load_data("data/house_data.csv")

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Evaluate
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # 5. Save predictions
    save_predictions(y_test, y_pred)

    # 6. Visualize
    plot_regression_line(model, X_test, y_test, feature_index=0,
                         feature_name="Size (sqft)")
    plot_residuals(y_test, y_pred)

    print("\n🎉 Pipeline complete. Check the outputs/ folder.")


if __name__ == "__main__":
    main()