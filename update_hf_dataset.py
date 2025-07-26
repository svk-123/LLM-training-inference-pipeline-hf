import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get token from environment variable
hf_username=os.getenv("HF_USERNAME")
hf_token = os.getenv("HF_TOKEN")

# Load your data (replace with your actual DataFrames)
train_df = pd.read_csv("./data/df_train.csv")  # Replace with your train DataFrame
test_df = pd.read_csv("./data/df_test.csv")    # Replace with your test DataFrame

train_df=train_df.iloc[:100]
test_df=test_df.iloc[:100]

# Convert to Hugging Face datasets and upload
dataset_dict = DatasetDict({
  "train": Dataset.from_pandas(train_df),
  "test": Dataset.from_pandas(test_df)
})

# Upload to Hugging Face Hub
dataset_dict.push_to_hub(f"{hf_username}/svg-code-generation", token=hf_token)

print("Dataset uploaded successfully!")