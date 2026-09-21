from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os


# --- CONFIGURATION ---
# The choice name 'visit-with-us-dataset'
repo_id = "BenBlay/visit-with-us-dataset"
repo_type = "dataset"

# Fetching token from environment variables
hf_token = os.getenv("HF_TOKEN")

# Initialize API client
api = HfApi(token=hf_token)

def register_data():
    """
    Checks for existence of a Hugging Face dataset repository,
    create it if missing, and upload the local data folder.
    """

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Dataset repo '{repo_id}' already exists. Preparing upload...")
    except RepositoryNotFoundError:

    # Step 2: Create the space if not found
        print(f"Repo '{repo_id}' not found. Creating new repository...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=hf_token)
        print(f"Repo '{repo_id}' created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Step 3: Upload the contents of the required subfolder 'data'
    # Per our requirement, we upload from 'tourism_project/data'
    print(f"Uploading files from 'tourism_project/data' to Hugging Face...")
    api.upload_folder(
        folder_path="tourism_project/data",
        repo_id=repo_id,
        repo_type=repo_type,
        token=hf_token
    )
    print(f'"Data Registration Complete! Data is now live at: https://huggingface.co/datasets/{repo_id}"')

if __name__ == "__main__":
    register_data()
