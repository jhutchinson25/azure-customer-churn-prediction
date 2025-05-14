import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from preprocess import build_preprocessor


def main():

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.3)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    # data preprocessing
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    churn_df = pd.read_csv(args.data).drop(columns='customerID')
    churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='coerce').fillna(0)
    churn_df['Churn'] = (churn_df['Churn'] == 'Yes') * 1
    print(churn_df)

    mlflow.log_metric("num_samples", churn_df.shape[0])
    mlflow.log_metric("num_features", churn_df.shape[1] - 1)

    train_df, test_df = train_test_split(
        churn_df,
        test_size=args.test_train_ratio,
    )

    # train model
    y_train = train_df.pop("Churn")

    X_train = train_df

    # Extracting the label column
    y_test = test_df.pop("Churn")

    X_test = test_df

    print(f"Training with data of shape {X_train.shape}")

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate
    )

    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", clf)
    ])

    pipeline.fit(X_train, y_train)

    # Save entire pipeline (includes preprocessing + model)
    y_pred = pipeline.predict(X_test)
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # save and register model
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )

    mlflow.end_run()


if __name__ == "__main__":
    main()
