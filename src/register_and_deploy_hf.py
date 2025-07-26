import os
import mlflow
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import shutil

def setup():
    """Load environment variables and setup MLflow"""
    load_dotenv()
    
    # Get tokens from .env file
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_username = os.getenv("HUGGINGFACE_USERNAME")
    
    if not hf_token or not hf_username:
        print("Missing HuggingFace credentials in .env file")
        print("Add these to your .env file:")
        print("HUGGINGFACE_TOKEN=your_token_here")
        print("HUGGINGFACE_USERNAME=your_username_here")
        return None, None
    
    # Setup MLflow
    mlflow.set_tracking_uri("file://./mlruns")
    client = MlflowClient()
    
    return hf_token, hf_username

def find_best_model():
    """Find the best trained model from MLflow experiments"""
    client = MlflowClient()
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name("unsloth-lora-experiments")
        if not experiment:
            print("No experiment found. Run training first!")
            return None
        
        # Get all runs, sorted by eval_loss (lower is better)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.eval_loss ASC"]
        )
        
        if not runs:
            print("No training runs found!")
            return None
        
        best_run = runs[0]
        run_name = best_run.data.tags.get('mlflow.runName', 'unknown')
        eval_loss = best_run.data.metrics.get('eval_loss', 'N/A')
        
        print(f"Best model found:")
        print(f"   Run: {run_name}")
        print(f"   Eval Loss: {eval_loss}")
        
        return best_run, run_name
        
    except Exception as e:
        print(f"Error finding best model: {e}")
        return None

def register_in_mlflow(run_id, run_name):
    """Register the best model in MLflow Model Registry"""
    try:
        model_name = "svg-code-generator"
        
        # Find the local model path
        local_model_path = f"./models/lora/{run_name}"
        
        if not os.path.exists(local_model_path):
            print(f"Model not found at: {local_model_path}")
            return None, None
        
        # Register model
        model_uri = f"file://{os.path.abspath(local_model_path)}"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Promote to Production
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        
        print(f"Model registered in MLflow:")
        print(f"   Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Stage: Production")
        
        return model_name, local_model_path
        
    except Exception as e:
        print(f"Error registering model: {e}")
        return None, None

def create_model_card(run_name):
    """Create README.md for HuggingFace"""
    return f"""# SVG Code Generator

This model generates SVG code from text descriptions.

## Model Details
- **Base Model**: Qwen3-4B
- **Training**: LoRA fine-tuning with Unsloth
- **Dataset**: vinoku89/svg-code-generation
- **Training Run**: {run_name}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("your-username/svg-code-generator")
model = AutoModelForCausalLM.from_pretrained("your-username/svg-code-generator")

# Generate SVG
prompt = "Generate a SVG code for the given input: red circle"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
"""