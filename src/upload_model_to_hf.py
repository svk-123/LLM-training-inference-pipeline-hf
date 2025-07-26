import os
import mlflow
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import tempfile

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
        
        # Handle different model source formats
        if model_source.startswith("file://"):
            local_model_path = model_source.replace("file://", "")
        elif model_source.startswith("models:/"):
            # Download the model from MLflow registry to a temporary location
            import tempfile
            temp_dir = tempfile.mkdtemp()
            local_model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_source,
                dst_path=temp_dir
            )
            print(f"   Downloaded model to: {local_model_path}")
        else:
            # Fallback: try to download using the run_id
            try:
                run = client.get_run(latest_version.run_id)
                artifact_uri = run.info.artifact_uri
                model_artifact_path = f"{artifact_uri}/model"
                
                import tempfile
                temp_dir = tempfile.mkdtemp()
                local_model_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=model_artifact_path,
                    dst_path=temp_dir
                )
                print(f"   Downloaded model from run artifacts to: {local_model_path}")
            except Exception as e:
                print(f"   Failed to download from run artifacts: {e}")
                return None, None
        
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
base_model: qwen3-0.6B
tags:
- code-generation
- svg
- fine-tuned
- fp16
- vllm
- merged
language:
- en
pipeline_tag: text-generation
library_name: transformers
model_type: qwen
inference: true
torch_dtype: float16
widget:
- example_title: "Simple Circle"
  text: "Create a red circle"
- example_title: "Rectangle with Border"
  text: "Draw a blue rectangle with black border"
- example_title: "Complex Shape"
  text: "Generate a star with 5 points in yellow"
---

# SVG Code Generator

This is a fine-tuned model for generating SVG code from natural language descriptions. The model has been merged with the base model weights and optimized in fp16 format.

## Model Details

- **Model Name**: {run_name}
- **Base Model**: qwen3-0.6B
- **Training Method**: Fine-tuning with merged weights
- **Task**: Text-to-SVG code generation
- **Model Type**: Merged Qwen model
- **Precision**: fp16
- **Library**: Transformers, vLLM compatible
- **Format**: Merged model (not adapter-based)

## Usage

### With Transformers

Load the model directly using the transformers library:

```python
# Load base model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("vinoku89/svg-code-generator")
model = AutoModelForCausalLM.from_pretrained("vinoku89/svg-code-generator")


# Generate SVG code
prompt = "Create a blue circle with radius 50"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with parameters
outputs = model.generate(
    **inputs, 
    max_length=200,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decode the generated SVG code
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
svg_code = generated_text[len(prompt):].strip()

print("Generated SVG:")
print(svg_code)
```

### With vLLM

This model supports vLLM for high-performance inference in fp16 format.

## Training Data

The model was trained on SVG code generation tasks with natural language descriptions.

## Intended Use

This model is designed to generate SVG code from text descriptions for educational and creative purposes.

## Limitations

- Generated SVG may require validation
- Performance depends on prompt clarity
- Limited to SVG syntax and features seen during training

## Model Performance

The model has been fine-tuned specifically for SVG generation tasks with merged weights for optimal performance.

## Technical Details

- **Precision**: fp16 for memory efficiency
- **Compatibility**: vLLM supported for high-throughput inference
- **Architecture**: Merged fine-tuned weights (no adapters required)
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
        
        # Use temporary file to avoid affecting local git repo
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_file.write(model_card_content)
            temp_readme_path = temp_file.name
        
        print("Uploading model files...")
        
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_name,
            token=hf_token,
            ignore_patterns=["*.git*", "*.DS_Store*", "__pycache__*"]
        )
        
        api.upload_file(
            path_or_fileobj=temp_readme_path,
            path_in_repo="README.md", 
            repo_id=repo_name,
            token=hf_token
        )
        
        # Clean up temporary file
        os.remove(temp_readme_path)
        
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