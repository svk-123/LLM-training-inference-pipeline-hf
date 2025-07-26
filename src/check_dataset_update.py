import os
import json
from huggingface_hub import HfApi
from datetime import datetime

def check_dataset_update():
    api = HfApi(token=os.getenv("HF_TOKEN"))
    dataset_name = "vinoku89/svg-code-generation"  # Just the repo ID, not full URL
    
    # Get dataset info
    dataset_info = api.dataset_info(dataset_name)
    last_modified = dataset_info.last_modified
    
    # Check if we have a stored timestamp
    timestamp_file = "last_dataset_check.json"
    
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            data = json.load(f)
            last_check = datetime.fromisoformat(data['last_modified'])
            
        if last_modified > last_check:
            should_retrain = True
            print("Dataset has been updated. Triggering retrain.")
        else:
            should_retrain = False
            print("No dataset updates found.")
    else:
        # First run
        should_retrain = True
        print("First run. Triggering retrain.")
    
    # Update timestamp
    with open(timestamp_file, 'w') as f:
        json.dump({
            'last_modified': last_modified.isoformat(),
            'checked_at': datetime.now().isoformat()
        }, f)
    
    # Set GitHub Actions output
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"should_retrain={str(should_retrain).lower()}\n")
    else:
        # For local testing
        print(f"should_retrain={str(should_retrain).lower()}")

if __name__ == "__main__":
    check_dataset_update()