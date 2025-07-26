import os
import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

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
            order_by=["metrics.loss ASC"]
        )
        
        if not runs:
            print("No training runs found!")
            return None
        
        best_run = runs[0]
        run_name = best_run.data.tags.get('mlflow.runName', 'unknown')
        eval_loss = best_run.data.metrics.get('loss', 'N/A')
        
        print(f"Best model found:")
        print(f"   Run: {run_name}")
        print(f"   Eval Loss: {eval_loss}")
        
        return best_run, run_name
        
    except Exception as e:
        print(f"Error finding best model: {e}")
        return None

def load_model_from_path(model_path):
    """Load model and tokenizer from local path"""
    try:
        print(f"Loading model from: {model_path}")
        
        # Try loading with Unsloth first
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            print("Model loaded successfully with Unsloth")
            return model, tokenizer
        except Exception as e:
            print(f"Unsloth loading failed: {e}")
            print("Trying with transformers...")
            
            # Fallback to transformers
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            print("Model loaded successfully with transformers")
            return model, tokenizer
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def register_in_mlflow(run_id, run_name):
    """Register the best model in MLflow Model Registry"""
    try:
        model_name = "svg-code-generator"
        
        # Find the local model path
        local_model_path = f"./models/lora/{run_name}"
        
        if not os.path.exists(local_model_path):
            print(f"Model not found at: {local_model_path}")
            return None, None
        
        # Check if model artifact exists in the run
        client = MlflowClient()
        try:
            artifacts = client.list_artifacts(run_id)
            model_artifact_exists = any(artifact.path == "model" for artifact in artifacts)
            
            if not model_artifact_exists:
                print("Model not found in run artifacts, logging model...")
                
                # Load the actual model and tokenizer
                model, tokenizer = load_model_from_path(local_model_path)
                if model is None or tokenizer is None:
                    print("Failed to load model, cannot register")
                    return None, None
                
                # Log the model to the existing run
                with mlflow.start_run(run_id=run_id):
                    model_info = mlflow.transformers.log_model(
                        transformers_model={
                            "model": model,
                            "tokenizer": tokenizer
                        },
                        artifact_path="model",
                        task="text-generation"
                    )
                    print(f"Model logged to run: {run_id}")
                
        except Exception as e:
            print(f"Error checking/logging model artifacts: {e}")
            return None, None
        
        # Register model using run URI
        model_uri = f"runs:/{run_id}/model"
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Use aliases instead of stages
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )
        
        # Add tags
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="validation_status",
            value="approved"
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="run_name",
            value=run_name
        )
        
        print(f"Model registered in MLflow:")
        print(f"   Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Alias: staging")
        print(f"   Run ID: {run_id}")
        
        return model_name, local_model_path
        
    except Exception as e:
        print(f"Error registering model: {e}")
        return None, None

def main():
    """Main deployment pipeline"""
    print("Starting model registration...")
    print("=" * 50)
       
    # Step 1: Find best model
    print("\nStep 1: Finding best model from MLflow...")
    result = find_best_model()
    if not result:
        print("No model found to register")
        exit(1)
    
    best_run, run_name = result
    print("Best model identified")
    
    # Step 2: Register in MLflow
    print("\nStep 2: Registering model in MLflow...")
    model_name, local_model_path = register_in_mlflow(best_run.info.run_id, run_name)
    if not local_model_path:
        print("Model registration failed")
        exit(1)
    
    print("Model registered in MLflow successfully")
    print("=" * 50)

if __name__ == "__main__":
    main()