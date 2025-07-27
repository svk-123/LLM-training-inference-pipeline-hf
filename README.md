# LLM Training & Inference Pipeline with Hugging Face

An end-to-end automated pipeline for training, tracking (MLflow), and deploying LLMs on Hugging Face Hub with GitHub Actions automation.


## Quick Setup

### 1. Environment Setup
```bash
# Initialize project
uv init
uv venv .venv --python=python3.11
source .venv/bin/activate

# Pin Python version (if required)
uv python pin python3.11

# Install dependencies
uv sync
```

### 2. Configuration
```.env
HF_USERNAME="your-huggingface-username"
HF_TOKEN="your-huggingface-token"
```

### 3. Start MLflow Server
```bash
# Start MLflow tracking server
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
# Server will be available at: http://localhost:5000
```

### 4.GitHub Secrets Setup
```
Add these secrets to your GitHub repository:
HF_TOKEN: Your HuggingFace access token
```

### 5. Manual Training
```bash
# Train model manually
uv run python ./src/train_llm.py

# Register in MLflow
uv run python ./src/register_model_mlflow.py

# Upload to HuggingFace
uv run python ./src/upload_model_to_hf.py
```

### 6. Automated Pipeline
- Automated Pipeline
- The GitHub Actions workflow runs automatically:
- Scheduled: Every 6 hours
- Manual: Via GitHub Actions "Run workflow" button
- Checks: Dataset updates before training
- Deploys: Automatically to HuggingFace Hub


### 7. Model Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model from HuggingFace
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

# Extract generated SVG
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
svg_code = generated_text[len(prompt):].strip()

print("Generated SVG:")
print(svg_code)
```

### 8. Deployment on HF Space (WIP)
- The trained model is auto-deployed in the following HF Space.
- https://huggingface.co/spaces/vinoku89/svg-code-generator
