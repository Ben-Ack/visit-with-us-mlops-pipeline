import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --- CONFIGURATION ---
hf_user = "BenBlay" # HF User ID
dataset_repo = f"{hf_user}/visit-with-us-dataset"
model_repo = f"{hf_user}/visit-with-us-model"
local_build_path = "tourism_project/model_building"
token = os.getenv("HF_TOKEN")

# 1. LOAD DATA FROM HUGGING FACE
print("Loading data from Hugging Face...")
Xtrain = pd.read_csv(f"https://huggingface.co/datasets/{dataset_repo}/raw/main/Xtrain.csv")
Xtest = pd.read_csv(f"https://huggingface.co/datasets/{dataset_repo}/raw/main/Xtest.csv")
ytrain = pd.read_csv(f"https://huggingface.co/datasets/{dataset_repo}/raw/main/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"https://huggingface.co/datasets/{dataset_repo}/raw/main/ytest.csv").values.ravel()

# 2. DEFINE FEATURES (Based on Tourism Data Dictionary)
numeric_features = [
    'Age',                      # Age of the customer
    'CityTier',                 # The city category
    'DurationOfPitch',          # Duration of the sales pitch delivered
    'NumberOfPersonVisiting',   # Number of people accompanying customer
    'NumberOfFollowups',        # Number of after sale follow-ups
    'PreferredPropertyStar',    # Preferred hotel rating by the customer
    'NumberOfTrips',            # Average number of customers' annual trips
    'Passport',                 # Whether the customer holds a valid passport
    'PitchSatisfactionScore',   # customer's satisfaction with the sales pitch
    'OwnCar',                   # Whether the customer owns a car
    'NumberOfChildrenVisiting', # Number of accompanying children below age 5
    'MonthlyIncome'             # Gross monthly income of the customer
]

categorical_features = [
    'TypeofContact',            # Method of customer inquiry
    'Occupation',               # Customer's occupation
    'Gender',                   # Gender of the customer
    'ProductPitched',           # Type of product pitched to the customer
    'MaritalStatus',            # Marital status of the customer
    'Designation'               # Customer's current designation
]

# 3. HANDLE CLASS IMBALANCE (Objective: Better targeting of customers)
# Calculating scale_pos_weight: count(negative) / count(positive)
neg, pos = np.bincount(ytrain)
class_weight = neg / pos

# 4. PREPROCESSING & PIPELINE
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

base_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
model_pipeline = make_pipeline(preprocessor, base_model)

# 5. EXPERIMENT TRACKING WITH MLFLOW
mlflow.set_experiment("Tourism_Package_Promotion")

with mlflow.start_run(run_name="XGBoost_GridSearch_Tuning"):
    # Define hyperparameter grid for tuning
    param_grid = {
        'xgbclassifier__n_estimators': [100, 150],
        'xgbclassifier__max_depth': [3, 5],
        'xgbclassifier__learning_rate': [0.05, 0.1],
        'xgbclassifier__colsample_bytree': [0.5, 0.8]
    }

    # Hyperparameter tuning
    print("Tuning model with GridSearchCV...")
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Evaluating
    y_pred = best_model.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    # Log to MLflow ('Log all the tuned parameters')
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(best_model, "tourism_xgboost_model")

    print(f"Model Tuned. Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

    # 6. SAVE BEST MODEL LOCALLY
    os.makedirs(local_build_path, exist_ok=True)
    model_file = f"{local_build_path}/best_tourism_model.joblib"
    joblib.dump(best_model, model_file)
    print(f"Model saved locally to {model_file}")

# 7. REGISTER BEST MODEL IN HUGGING FACE MODEL HUB
api = HfApi(token=token)

try:
    api.repo_info(repo_id=model_repo, repo_type="model")
    print(f"Model Hub '{model_repo}' exists.")
except RepositoryNotFoundError:
    print(f"Creating new Model Hub repository: {model_repo}")
    create_repo(repo_id=model_repo, repo_type="model", private=False, token=token)

print(f"Registering best model to Hugging Face Model Hub...")
api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="best_tourism_model.joblib",
    repo_id=model_repo,
    repo_type="model"
)

print(f"Deployment Ready! Model registered at: https://huggingface.co/models/{model_repo}")
