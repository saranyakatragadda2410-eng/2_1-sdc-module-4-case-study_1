import matplotlib.pyplot as plt
import numpy as np

def plot_regression_line(model, X_test, y_test, feature_index=0,
                         feature_name="Size (sqft)", path="outputs/regression_plot.png"):
    """Plot regression line for the primary feature vs price."""
    X_axis = X_test[:, feature_index]
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_axis, y_test, color="steelblue", alpha=0.7, label="Actual")
    plt.scatter(X_axis, y_pred, color="darkorange", alpha=0.7, label="Predicted")

    # Best fit line (based on this 1D slice)
    z = np.polyfit(X_axis, y_pred, 1)
    p = np.poly1d(z)
    xs = np.linspace(X_axis.min(), X_axis.max(), 100)
    plt.plot(xs, p(xs), color="red", linewidth=2, label="Regression line")

    plt.xlabel(feature_name + " (standardized)")
    plt.ylabel("Price (USD)")
    plt.title("House Price Regression — Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"🖼️  Regression plot saved: {path}")


def plot_residuals(y_test, y_pred, path="outputs/residuals_plot.png"):
    """Residual analysis plot."""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, color="purple", alpha=0.7)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")

    # Histogram of residuals
    axes[1].hist(residuals, bins=15, color="teal", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Residuals")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"🖼️  Residual plot saved: {path}")