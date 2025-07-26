import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import spaces

# Load model
tokenizer = AutoTokenizer.from_pretrained("vinoku89/qwen3-4B-svg-code-gen")
model = AutoModelForCausalLM.from_pretrained("vinoku89/qwen3-4B-svg-code-gen")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@spaces.GPU
def generate_svg(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    svg_code = generated_text[len(prompt):].strip()
    svg_display = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'>{svg_code}</svg>"
    
    return svg_code, svg_display

gradio_app = gr.Interface(
    fn=generate_svg,
    inputs=gr.Textbox(lines=2, placeholder="Describe the SVG you want..."),
    outputs=[
        gr.Code(label="Generated SVG Code", language="html"),
        gr.HTML(label="SVG Preview")
    ],
    title="SVG Code Generator",
    description="Generate SVG code from natural language using a fine-tuned LLM."
)

if __name__ == "__main__":
    gradio_app.launch()
    
