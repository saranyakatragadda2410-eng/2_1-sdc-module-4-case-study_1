from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train) -> LinearRegression:
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model trained successfully")
    print(f"   Intercept: {model.intercept_:.2f}")
    print(f"   Coefficients: {model.coef_}")
    return model