import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model with standard regression metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    return metrics, y_pred

def save_predictions(y_test, y_pred, path="outputs/predictions.csv"):
    """Save actual vs predicted values for inspection."""
    out = pd.DataFrame({"actual_price": y_test.values, "predicted_price": y_pred})
    out["residual"] = out["actual_price"] - out["predicted_price"]
    out.to_csv(path, index=False)
    print(f"💾 Predictions saved to {path}")
    return out