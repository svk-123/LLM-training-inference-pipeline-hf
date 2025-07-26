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
