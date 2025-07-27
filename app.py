import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import spaces
import xml.etree.ElementTree as ET
import re

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Load model with memory optimizations
model_path = "/home/vino/ML_Projects/Drawing_with_LLMs/lora/Qwen3_4B_lora_fp16_r256_s60000_e2_1_msl2048"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True  # Add this if needed for custom models
)

def validate_svg(svg_content):
    """
    Validate if SVG content is properly formatted and renderable
    """
    try:
        # Clean up the SVG content
        svg_content = svg_content.strip()
        
        # If it doesn't start with <svg, try to extract SVG content
        if not svg_content.startswith('<svg'):
            # Look for SVG tags in the content
            svg_match = re.search(r'<svg[^>]*>.*?</svg>', svg_content, re.DOTALL | re.IGNORECASE)
            if svg_match:
                svg_content = svg_match.group(0)
            else:
                # If no complete SVG found, wrap content in SVG tags
                if any(tag in svg_content.lower() for tag in ['<circle', '<rect', '<path', '<line', '<polygon', '<ellipse', '<text']):
                    svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="250" height="250">{svg_content}</svg>'
                else:
                    raise ValueError("No valid SVG elements found")
        
        # Parse XML to validate structure
        ET.fromstring(svg_content)
        
        return True, svg_content
        
    except ET.ParseError as e:
        return False, f"XML Parse Error: {str(e)}"
    except Exception as e:
        return False, f"Validation Error: {str(e)}"

@spaces.GPU(duration=60)  # Add duration limit
def generate_svg(prompt):
    # Clear cache before generation
    torch.cuda.empty_cache()
    
    # Format the prompt using Alpaca template
    instruction = "Generate SVG code based on the given description."
    formatted_prompt = alpaca_prompt.format(
        instruction,
        prompt,
        ""  # Empty response - model will fill this
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move inputs to the same device as model
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():  # Disable gradient computation to save memory
        outputs = model.generate(
            **inputs,
            max_length=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=512  # Limit new tokens instead of total length
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (after "### Response:")
    response_start = generated_text.find("### Response:")
    if response_start != -1:
        svg_code = generated_text[response_start + len("### Response:"):].strip()
    else:
        # Fallback: remove the original formatted prompt
        svg_code = generated_text[len(formatted_prompt):].strip()
    
    # Validate SVG
    is_valid, result = validate_svg(svg_code)
    
    if is_valid:
        # SVG is valid
        validated_svg = result
        # Ensure the SVG has proper dimensions for display (keep moderate size)
        if 'width=' not in validated_svg or 'height=' not in validated_svg:
            validated_svg = validated_svg.replace('<svg', '<svg width="250" height="250"', 1)
        svg_display = validated_svg
    else:
        # SVG is invalid, show error message
        svg_display = f"""
        <div style="width: 250px; height: 200px; border: 2px dashed #ff6b6b; 
                    display: flex; align-items: center; justify-content: center; 
                    background-color: #fff5f5; border-radius: 8px; padding: 15px; 
                    text-align: center; color: #e03131; font-family: Arial, sans-serif;">
            <div>
                <h4 style="margin: 0 0 8px 0; color: #e03131;">🚫 Preview Not Available</h4>
                <p style="margin: 0; font-size: 12px;">Generated SVG contains errors:<br>
                <em style="font-size: 11px;">{result}</em></p>
            </div>
        </div>
        """
    
    # Clear cache after generation
    torch.cuda.empty_cache()
    
    return svg_code, svg_display

# Minimal CSS for slightly larger HTML preview only
custom_css = """
div[data-testid="HTML"] {
    min-height: 320px !important;
}
"""

gradio_app = gr.Interface(
    fn=generate_svg,
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Describe the SVG you want (e.g., 'a red circle with blue border')..."
    ),
    outputs=[
        gr.Code(label="Generated SVG Code", language="html"),
        gr.HTML(label="SVG Preview")
    ],
    title="SVG Code Generator",
    description="Generate SVG code from natural language using a fine-tuned LLM.",
    css=custom_css
)

if __name__ == "__main__":
    gradio_app.launch()