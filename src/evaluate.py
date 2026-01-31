import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_data, split_data

model_uri = "runs:/{}/model".format(mlflow.active_run().info.run_id)

df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

model = mlflow.sklearn.load_model(model_uri)

test_preds = model.predict(X_test)

mlflow.log_metric("test_rmse",
                  mean_squared_error(y_test, test_preds, squared=False))
mlflow.log_metric("test_r2", r2_score(y_test, test_preds))

print("Test evaluation logged")
