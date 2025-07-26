import pandas as pd
from datasets import Dataset, DatasetDict
import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import mlflow
import torch

def setup_environment():
    """Setup environment variables and CUDA settings"""
    load_dotenv()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    return os.getenv("HUGGINGFACE_TOKEN")

def setup_mlflow():
    """Setup MLflow tracking with proper configuration"""
    # Create MLflow directory if it doesn't exist
    mlflow_dir = "./mlruns"
    os.makedirs(mlflow_dir, exist_ok=True)
    
    # Set tracking URI to local directory
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")
    
    # Set experiment
    experiment_name = "unsloth-lora-experiments"
    try:
        experiment = mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment created/found: {experiment_name}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        return experiment
    except Exception as e:
        print(f"Error setting up MLflow: {e}")
        return None

def load_training_data_from_hf(dataset_name="vinoku89/svg-code-generation"):
    """Load training data from HuggingFace dataset"""
    splits = {
        'train': 'data/train-00000-of-00001.parquet', 
        'test': 'data/test-00000-of-00001.parquet'
    }
    
    df_train = pd.read_parquet(f"hf://datasets/{dataset_name}/" + splits["train"])
    df_test = pd.read_parquet(f"hf://datasets/{dataset_name}/" + splits["test"])
    
    print("Train dataset size:",df_train.shape)
    print("Test dataset size:",df_test.shape)

    # Ensure correct column names
    required_columns = ['description', 'clean_svg']
    df_train = df_train[required_columns]
    df_test = df_test[required_columns]
    
    return df_train, df_test

def load_model(base_model="Qwen3-4B", max_seq_length=2048, model_path=None):
    """Load and configure the language model"""
    if model_path:
        model_name = model_path
    else:
        model_name = f"unsloth/{base_model}"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        load_in_8bit=False
    )
    
    return model, tokenizer

def setup_lora_model(model, rank=256):
    """Setup LoRA configuration for the model"""
    return FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123
    )

def formatting_prompts_func(examples, tokenizer):
    """Format the examples for training"""
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""
    EOS_TOKEN = tokenizer.eos_token
    topics = examples["description"]
    svgs = examples["clean_svg"]
    texts = []
    
    for topic, svg_code in zip(topics, svgs):
        text = alpaca_prompt.format("Generate a SVG code for the given input:", topic, svg_code) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

def prepare_datasets(df_train, df_test, tokenizer):
    """Convert dataframes to HuggingFace datasets and format them"""
    dataset_train = Dataset.from_pandas(df_train)
    dataset_train = dataset_train.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
    
    dataset_test = Dataset.from_pandas(df_test)
    dataset_test = dataset_test.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
    
    return dataset_train, dataset_test

def train_model(model, tokenizer, dataset_train, dataset_test, epochs=1, max_seq_length=2048, run_name="training_run"):
    """Train the model using SFTTrainer"""
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=epochs,
            max_steps=-1,
            learning_rate=5e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=123,
            eval_strategy="steps",
            eval_steps=10,
            output_dir="outputs",
            report_to="mlflow",
            run_name=run_name,  # Add run name here too
        ),
    )
    
    # Log additional parameters to MLflow
    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_params({
            "model_name": "Qwen3-4B",
            "max_seq_length": max_seq_length,
            "epochs": epochs,
            "learning_rate": 5e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
        })
        
        print(f"MLflow run started: {run.info.run_id}")
        print(f"MLflow run name: {run_name}")
        
        trainer.train()
        
        print(f"MLflow run completed: {run.info.run_id}")
    
    return trainer

def save_model(model, tokenizer, run_name):
    """Save the trained model"""
    dir_path = f"./models/lora/{run_name}"
    os.makedirs(dir_path, exist_ok=True)
    model.save_pretrained_merged(dir_path, tokenizer, save_method="merged_16bit")
    return dir_path

def main():
    """Main function to run the entire pipeline"""
    # Configuration
    base_model = "Qwen3-4B"
    max_seq_length = 2048
    rank = 64
    epochs = 5
    dataset_name = "vinoku89/svg-code-generation"
    
    # Create run name
    train_parameters = f"_lora_fp16_r{rank}_e{epochs}_msl{max_seq_length}"
    run_name = base_model.replace(".", "").replace("-", "_") + train_parameters
    
    # Setup
    hf_token = setup_environment()
    
    # Setup MLflow
    experiment = setup_mlflow()
    if experiment is None:
        print("Warning: MLflow setup failed, continuing without MLflow tracking")
    
    # Load model
    model, tokenizer = load_model(base_model, max_seq_length)
    model = setup_lora_model(model, rank)
    
    # Load and prepare data
    df_train, df_test = load_training_data_from_hf(dataset_name)
    dataset_train, dataset_test = prepare_datasets(df_train, df_test, tokenizer)
    
    # Train model
    trainer = train_model(model, tokenizer, dataset_train, dataset_test, epochs, max_seq_length, run_name)
    
    # Save model
    model_path = save_model(model, tokenizer, run_name)
    print(f"Model saved to: {model_path}")
    
    # Show MLflow UI command
    print("\n" + "="*50)
    print("To view MLflow UI, run this command in another terminal:")
    print("mlflow ui --backend-store-uri ./mlruns")
    print("Then open: http://localhost:5000")
    print("="*50)

if __name__ == "__main__":
    main()