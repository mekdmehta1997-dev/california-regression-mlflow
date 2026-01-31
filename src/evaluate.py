import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_data, split_data

# ✅ Load run_id
with open("run_id.txt", "r") as f:
    run_id = f.read().strip()

df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

test_preds = model.predict(X_test)

with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("test_rmse", np.sqrt(mean_squared_error(y_test, test_preds)))
    mlflow.log_metric("test_r2", r2_score(y_test, test_preds))

print("Test metrics logged successfully")
