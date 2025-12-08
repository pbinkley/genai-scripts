# based on https://huggingface.co/allenai/olmOCR-2-7B-1025

import torch
import base64
import urllib.request
import hashlib
import datetime

from io import BytesIO
import json  # Import the json module
import os
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pathlib import Path

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
import PyPDF2 # Import PyPDF2 to get PDF page count

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

if not os.path.exists("output"):
  os.makedirs("output")

pdf_filepath = args.pdf_filepath

# Get PDF page count using PyPDF2
with open(pdf_filepath, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
print(f"{pdf_filepath} has {num_pages} pages.")

num_pages = 1

page_results = [] # List to store PageResult objects (like in pipeline.py)
char_spans = [] # List to store character spans
document_text = "" # Accumulate document text
current_char_pos = 0 # Track character position (like in pipeline.py)

for page_num in range(1, num_pages + 1): # Loop through all pages
    print(f"\n--- Processing Page {page_num} ---")
  
    # Render page 1 to an image
    image_base64 = render_pdf_to_base64png(args.pdf_filepath, page_num, target_longest_image_dim=1288)
    
    
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
    for chunk in text_output:
      document_text += chunk.strip() + ("\n" if page_num < num_pages else "") # Accumulate text
#    import pdb; pdb.set_trace() 
    # print(text_output)
    # ['---\nprimary_language: en\nis_rotation_valid: True\nrotation_correction: 0\nis_table: False\nis_diagram: False\n---\nolmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models\n\nJake Poz']

print("\n\n--- Generation Complete for All Pages ---")

# --- Build Dolma Document JSON (Corrected pdf_page_numbers format) ---
metadata = { # Simplified metadata - just source file and page count
    "Source-File": pdf_filepath,
    "pdf-total-pages": num_pages,
}
id_ = hashlib.sha1(document_text.encode()).hexdigest() # Generate document ID
dolma_doc = { # Create Dolma document JSON
    "id": id_,
    "text": document_text.strip(), # Document text (all pages combined)
    "source": args.source,
    "added": datetime.datetime.now().strftime("%Y-%m-%d"),
    "created": datetime.datetime.now().strftime("%Y-%m-%d"),
    "metadata": metadata,
    "attributes": {
        "pdf_page_numbers": [[span[0], span[1], res.page_num] for span, res in zip(char_spans, page_results)]
    }, 
}

print("\nDolma Document JSON (Corrected pdf_page_numbers format):")
print(dolma_doc) # Print the Dolma document JSON

# Save to JSON Lines file (single Dolma document JSON object)
output_file = args.output
if output_file is None:
    output_file = "./output/" + Path(pdf_filepath).stem + ".json"
    output_file_text = "./output/" + Path(pdf_filepath).stem + ".txt"

try:
    with open(output_file, "w") as f:
        json.dump(dolma_doc, f) # Write the single Dolma document JSON object
        f.write('\n') # Add newline (for JSON Lines format - single line file)
    print(f"\nGenerated text saved to {output_file} (JSON Lines format, single Dolma document, corrected pdf_page_numbers)")
    page_counter = 0
    with open(output_file_text, "w") as f:
        for page in dolma_doc['text'].splitlines():
            import pdb; pdb.set_trace() 
            page_counter += 1
            page_text = json.loads(page)['natural_text']
            page_words = page_text.split()
            # Parse each line as JSON
            try:
                f.write(f"--- page {page_counter} ----------\n\n{page_text}\n\n")
                print(f"Page {page_counter}: {' '.join(page_words[0:4])} ... {' '.join(page_words[-4:])}")
            except json.JSONDecodeError as e:
                print(f"Error parsing text: {page}")
                print(e)
except Exception as save_error:
    print(f"\nError saving output to file {output_file}:\n{save_error}")
