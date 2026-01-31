import mlflow
import mlflow.sklearn
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from data_loader import load_data, split_data
import mlflow
print("Tracking URI:", mlflow.get_tracking_uri())


mlflow.set_experiment("California-Housing-Regression")

df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    mlflow.log_metric("train_rmse", np.sqrt(mean_squared_error(y_train, train_preds)))
    mlflow.log_metric("val_rmse", np.sqrt(mean_squared_error(y_val, val_preds)))
    mlflow.log_metric("train_r2", r2_score(y_train, train_preds))
    mlflow.log_metric("val_r2", r2_score(y_val, val_preds))

    mlflow.sklearn.log_model(model, "model")

    # ✅ Save run_id for evaluation
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("Training & validation completed")

