from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient



def data_ingestion(**context):

    dataset_path = "/home/tahar/MLOPS_assignment_2/data/titanic.csv"

    df = pd.read_csv(dataset_path)

    # Print dataset shape
    print(f"Dataset shape: {df.shape}")

    # Count missing values
    missing_values = df.isnull().sum()

    print("Missing values per column:")
    print(missing_values)

    # Push dataset path using XCom
    context['ti'].xcom_push(
        key='dataset_path',
        value=dataset_path
    )


def data_validation(**context):
    ti = context['ti']

    dataset_path = ti.xcom_pull(
        key='dataset_path',
        task_ids='data_ingestion'
    )

    df = pd.read_csv(dataset_path)

    age_missing_pct = df['Age'].isnull().mean() * 100
    embarked_missing_pct = df['Embarked'].isnull().mean() * 100

    print(f"Age missing percentage: {age_missing_pct:.2f}%")
    print(f"Embarked missing percentage: {embarked_missing_pct:.2f}%")

    # Intentional failure demonstration
    if age_missing_pct > 30:
        raise ValueError("Age column has more than 30% missing values")

    if embarked_missing_pct > 30:
        raise ValueError("Embarked column has more than 30% missing values")

def handle_missing(**context):
    ti = context['ti']

    dataset_path = ti.xcom_pull(
        key='dataset_path',
        task_ids='data_ingestion'
    )

    df = pd.read_csv(dataset_path)

    # Fill missing Age with median
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fill missing Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    print("Missing values handled")

    processed_path = "/home/tahar/MLOPS_assignment_2/data/titanic_missing_handled.csv"

    df.to_csv(processed_path, index=False)

    ti.xcom_push(key="processed_data", value=processed_path)


def feature_engineering(**context):

    ti = context['ti']

    dataset_path = ti.xcom_pull(
        key='dataset_path',
        task_ids='data_ingestion'
    )

    df = pd.read_csv(dataset_path)

    # Create FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Create IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    print("Feature engineering completed")

    feature_path = "/home/tahar/MLOPS_assignment_2/data/titanic_features.csv"

    df.to_csv(feature_path, index=False)

    ti.xcom_push(key="feature_data", value=feature_path)


def encoding(**context):

    ti = context['ti']

    dataset_path = ti.xcom_pull(
        key='dataset_path',
        task_ids='data_ingestion'
    )

    df = pd.read_csv(dataset_path)

    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Encode Embarked
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Drop irrelevant columns
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

    print("Encoding completed")

    encoded_path = "/home/tahar/MLOPS_assignment_2/data/titanic_encoded.csv"

    df.to_csv(encoded_path, index=False)

    ti.xcom_push(key="encoded_data", value=encoded_path)


def train_model(**context):
    ti = context['ti']

    # Read hyperparameters from Airflow params
    params = context["params"]

    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 5)

    # Load encoded dataset
    encoded_path = "/home/tahar/MLOPS_assignment_2/data/titanic_encoded.csv"

    df = pd.read_csv(encoded_path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Set MLflow experiment
    mlflow.set_experiment("Titanic_Survival_Pipeline")

    # Start MLflow run
    with mlflow.start_run() as run:

        run_id = run.info.run_id

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log dataset size
        mlflow.log_param("dataset_rows", len(df))

        # Log trained model artifact
        mlflow.sklearn.log_model(model, "random_forest_model")

        print("Model training complete")

        # Pass run_id to evaluation task
        ti.xcom_push(key="run_id", value=run_id)


def evaluate_model(**context):

    ti = context['ti']

    run_id = ti.xcom_pull(
        key="run_id",
        task_ids="model_training"
    )

    params = context["params"]

    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 5)

    encoded_path = "/home/tahar/MLOPS_assignment_2/data/titanic_encoded.csv"

    df = pd.read_csv(encoded_path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    with mlflow.start_run(run_id=run_id):

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    ti.xcom_push(key="accuracy", value=accuracy)


def branching(**context):

    ti = context['ti']

    accuracy = ti.xcom_pull(
        key="accuracy",
        task_ids="model_evaluation"
    )

    print(f"Model accuracy: {accuracy}")

    if accuracy >= 0.80:
        return "register_model"
    else:
        return "reject_model"

def register_model(**context):
    ti = context['ti']

    run_id = ti.xcom_pull(
        key="run_id",
        task_ids="model_training"
    )

    model_uri = f"runs:/{run_id}/random_forest_model"

    mlflow.register_model(
        model_uri=model_uri,
        name="Titanic_Survival_Model"
    )

def reject_model(**context):
    ti = context['ti']

    accuracy = ti.xcom_pull(
        key="accuracy",
        task_ids="model_evaluation"
    )

    print(f"Model rejected due to low accuracy: {accuracy}")

with DAG(
    dag_id="titanic_ml_pipeline",
    start_date=datetime(2024,1,1),
    schedule=None,
    catchup=False,
    params={
        "n_estimators": 100,
        "max_depth": 5
    }
) as dag:

    start = EmptyOperator(task_id="start")

    ingestion = PythonOperator(
    task_id="data_ingestion",
    python_callable=data_ingestion
    )

    validation = PythonOperator(
    task_id="data_validation",
    python_callable=data_validation,
    retries=2,
    retry_delay=timedelta(seconds=10)
    )

    missing = PythonOperator(
        task_id="missing_value_handling",
        python_callable=handle_missing
    )

    feature = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering
    )

    encode = PythonOperator(
        task_id="encoding",
        python_callable=encoding
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=train_model
    )

    evaluate = PythonOperator(
        task_id="model_evaluation",
        python_callable=evaluate_model
    )

    branch = BranchPythonOperator(
    task_id="branching",
    python_callable=branching
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model
    )

    reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model
    )

    start >> ingestion >> validation

    validation >> [missing, feature]

    [missing, feature] >> encode >> train >> evaluate >> branch

    branch >> [register, reject]
