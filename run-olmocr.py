# based on https://huggingface.co/allenai/olmOCR-2-7B-1025

import torch
import base64
import urllib.request

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

import argparse

parser = argparse.ArgumentParser(prog="run-olmocr.py", description="Run olmocr OCR process on a pdf.")
parser.add_argument("--model", type=str, default="allenai/olmOCR-2-7B-1025", help="Model path")
parser.add_argument("pdf_filepath", help="Path of pdf file to read")
parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
parser.add_argument("--source", type=str, default="olmocr", help="Source, for Dolma doc")
parser.add_argument("--language", type=str, default="en", help="Primary language")
parser.add_argument("--output", type=str, default=None, help="Output file for Dolma doc")
#parser.add_argument("", type=, default=, help="")
#parser.add_argument("", type=, default=, help="")
#parser.add_argument("", type=, default=, help="")

args = parser.parse_args()

# Initialize the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("allenai/olmOCR-2-7B-1025", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # Keep device check
model.to(device)

# Grab a sample PDF
# urllib.request.urlretrieve("https://olmocr.allenai.org/papers/olmocr.pdf", "./paper.pdf")

# Render page 1 to an image
image_base64 = render_pdf_to_base64png(args.pdf_filepath, 1, target_longest_image_dim=1288)


# Build the full prompt
messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

# Apply the chat template and processor
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

inputs = processor(
    text=[text],
    images=[main_image],
    padding=True,
    return_tensors="pt",
)
inputs = {key: value.to(device) for (key, value) in inputs.items()}


# Generate the output
output = model.generate(
            **inputs,
            max_new_tokens = args.max_tokens,
            temperature = args.temperature,
            num_return_sequences=1,
            do_sample=True,
        )

# Decode the output
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(
    new_tokens, skip_special_tokens=True
)

print(text_output)
# ['---\nprimary_language: en\nis_rotation_valid: True\nrotation_correction: 0\nis_table: False\nis_diagram: False\n---\nolmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models\n\nJake Poz']
