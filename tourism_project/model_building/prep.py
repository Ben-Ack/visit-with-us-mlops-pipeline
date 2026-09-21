import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# --- CONFIGURATION ---
# Replace 'your-username' with your actual Hugging Face username
hf_user = "BenBlay"
repo_id = f"{hf_user}/visit-with-us-dataset"
local_data_path = "tourism_project/data"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# 1. LOAD DATASET
# Loading directly from the Hugging Face Dataset Hub registered in Step 1
DATASET_URL = f"https://huggingface.co/datasets/{repo_id}/raw/main/tourism.csv"
df = pd.read_csv(DATASET_URL)
print(f"Dataset loaded from {repo_id}")

# 2. DATA CLEANING
# Removing unnecessary columns
# CustomerID is just an index; Unnamed: 0 often appears in CSV exports
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True, errors='ignore')

# Filling missing numerical values with the median
for col in ['Age', 'DurationOfPitch', 'MonthlyIncome', 'NumberOfTrips']:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical/discrete values with the mode
for col in ['TypeofContact', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfChildrenVisiting']:
    df[col] = df[col].fillna(df[col].mode()[0])
# Fixing Typos (Analytical fix for 'Fe Male')
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

print("Data cleaning and typo correction complete.")

# 3. DEFINE FEATURES AND TARGET
target = 'ProdTaken'
X = df.drop(columns=[target])
y = df[target]

# 4. SPLIT DATASET (Rubric Requirement: 'Split into training and testing sets')
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Ensures the purchase ratio is the same in both sets
)

# 5. SAVE LOCALLY (Rubric Requirement: 'Save them locally' in subfolder)
os.makedirs(local_data_path, exist_ok=True)

Xtrain.to_csv(f"{local_data_path}/Xtrain.csv", index=False)
Xtest.to_csv(f"{local_data_path}/Xtest.csv", index=False)
ytrain.to_csv(f"{local_data_path}/ytrain.csv", index=False)
ytest.to_csv(f"{local_data_path}/ytest.csv", index=False)

print(f"Split files saved locally to {local_data_path}/")

# 6. UPLOAD TO HUGGING FACE ('Upload back to the HF data space')
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

print(f"Uploading processed splits to Hugging Face...")
for file_name in files:
    api.upload_file(
        path_or_fileobj=f"{local_data_path}/{file_name}",
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="dataset"
    )

print(f"Data Preparation Complete! Files available at: https://huggingface.co/datasets/{repo_id}")
