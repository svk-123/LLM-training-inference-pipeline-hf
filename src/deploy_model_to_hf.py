import os
import mlflow
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

def setup():
   load_dotenv()
   
   hf_token = os.getenv("HF_TOKEN")
   hf_username = os.getenv("HF_USERNAME")
   
   if not hf_token or not hf_username:
       print("Missing HuggingFace credentials in .env file")
       return None, None
   
   mlflow.set_tracking_uri("./mlruns")
   
   return hf_token, hf_username

def get_model_path_from_mlflow():
   try:
       client = MlflowClient()
       model_name = "svg-code-generator"
       
       model_versions = client.search_model_versions(f"name='{model_name}'")
       
       if not model_versions:
           print(f"No registered model found with name: {model_name}")
           return None, None
       
       latest_version = max(model_versions, key=lambda v: int(v.version))
       
       print(f"Latest registered model:")
       print(f"   Name: {model_name}")
       print(f"   Version: {latest_version.version}")
       print(f"   Run ID: {latest_version.run_id}")
       
       run_name = f'model_v{latest_version.version}'
       
       model_source = latest_version.source
       print(f"   Model Source: {model_source}")
       print(f"   Run Name: {run_name}")
       
       if model_source.startswith("file://"):
           local_model_path = model_source.replace("file://", "")
       else:
           local_model_path = f"./models/lora/{run_name}"
       
       if not os.path.exists(local_model_path):
           print(f"Model path does not exist: {local_model_path}")
           return None, None
       
       print(f"   Local Path: {local_model_path}")
       
       return local_model_path, run_name
       
   except Exception as e:
       print(f"Error getting model from MLflow: {e}")
       return None, None

def create_model_card(run_name):
   model_card = f"""---
license: apache-2.0
tags:
- code-generation
- svg
- lora
- fine-tuned
language:
- en
pipeline_tag: text-generation
---

# SVG Code Generator

This is a fine-tuned LoRA adapter for generating SVG code from natural language descriptions.

## Model Details

- **Model Name**: {run_name}
- **Base Model**: Fine-tuned language model
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Task**: Text-to-SVG code generation

## Usage

Load the model using the transformers library and PEFT for LoRA adapters. Use natural language prompts to generate SVG code.

## Training Data

The model was trained on SVG code generation tasks with natural language descriptions.

## Intended Use

This model is designed to generate SVG code from text descriptions for educational and creative purposes.

## Limitations

- Generated SVG may require validation
- Performance depends on prompt clarity
- Limited to SVG syntax and features seen during training

## Model Performance

The model has been fine-tuned specifically for SVG generation tasks and should be used within this domain for best results.
"""
   return model_card

def upload_to_huggingface(local_model_path, run_name, hf_token, hf_username):
   try:
       api = HfApi()
       repo_name = f"{hf_username}/svg-code-generator"
       
       print(f"Creating HuggingFace repository: {repo_name}")
       
       create_repo(repo_name, token=hf_token, exist_ok=True)
       
       print("Creating model card...")
       model_card_content = create_model_card(run_name)
       
       with open("README.md", "w") as f:
           f.write(model_card_content)
       
       print("Uploading model files...")
       
       api.upload_folder(
           folder_path=local_model_path,
           repo_id=repo_name,
           token=hf_token,
           ignore_patterns=["*.git*", "*.DS_Store*", "__pycache__*"]
       )
       
       api.upload_file(
           path_or_fileobj="README.md",
           path_in_repo="README.md", 
           repo_id=repo_name,
           token=hf_token
       )
       
       os.remove("README.md")
       
       print(f"SUCCESS: Model uploaded to: https://huggingface.co/{repo_name}")
       return repo_name
       
   except Exception as e:
       print(f"ERROR: {e}")
       return None

def main():
   print("Starting HuggingFace deployment...")
   
   hf_token, hf_username = setup()
   if not hf_token or not hf_username:
       return
   
   local_model_path, run_name = get_model_path_from_mlflow()
   if not local_model_path or not run_name:
       return
   
   repo_name = upload_to_huggingface(local_model_path, run_name, hf_token, hf_username)
   if repo_name:
       print(f"Model available at: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
   main()