import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_data, split_data

mlflow.set_experiment("California-Housing-Regression")

df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    mlflow.log_metric("train_rmse",
                      mean_squared_error(y_train, train_preds, squared=False))
    mlflow.log_metric("val_rmse",
                      mean_squared_error(y_val, val_preds, squared=False))

    mlflow.log_metric("train_r2", r2_score(y_train, train_preds))
    mlflow.log_metric("val_r2", r2_score(y_val, val_preds))

    mlflow.sklearn.log_model(model, "model")

print("Training & Validation complete")
