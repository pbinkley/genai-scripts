# from https://github.com/allenai/olmocr/issues/33#issuecomment-2726291140

import torch
import base64
import urllib.request
import json  # Import the json module
import os # Import os module for file path operations
import datetime
import hashlib

from io import BytesIO
from PIL import Image
from pathlib import Path

# Use mlx_vlm's load to load both model and processor
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image # <-- Import load_image from mlx_vlm.utils

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import PageResponse, build_finetuning_prompt # <--- PageResponse import (CORRECTED)
from olmocr.prompts.anchor import get_anchor_text
from olmocr.pipeline import PageResult # Import PageResult dataclass
from dataclasses import dataclass # Import dataclass (for PageResult definition)
from typing import Optional, List # Import Optional and List (for PageResult definition)
import PyPDF2 # Import PyPDF2 to get PDF page count

import argparse

parser = argparse.ArgumentParser(description="Run olmocr OCR process on a pdf.")
parser.add_argument("--model", type=str, default="mlx-community/olmOCR-7B-0225-preview-4bit", help="Model path")
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model and processor using mlx_vlm.load (like the example)
olmocr_model, olmocr_processor = load(args.model) # Load both model and processor from mlx_vlm
olmocr_config = olmocr_model.config # Get model config

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # Keep device check

# Grab a sample PDF (same as original)
pdf_filepath = args.pdf_filepath # Define filepath for clarity
#urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", pdf_filepath)

# Get PDF page count using PyPDF2
with open(pdf_filepath, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
print(f"{pdf_filepath} has {num_pages} pages.")

page_results = [] # List to store PageResult objects (like in pipeline.py)
char_spans = [] # List to store character spans
document_text = "" # Accumulate document text
current_char_pos = 0 # Track character position (like in pipeline.py)

for page_num in range(1, num_pages + 1): # Loop through all pages
    print(f"\n--- Processing Page {page_num} ---")

    # Render page to an image
    image_base64 = render_pdf_to_base64png(pdf_filepath, page_num, target_longest_image_dim=1024)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64))) # Load PIL Image

    # Build the prompt, using document metadata (same as original)
    anchor_text = get_anchor_text(pdf_filepath, page_num, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text) # Original dynamic prompt

    # Build messages
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    # print(f"apply_chat_template function: {apply_chat_template}")
    text_prompt = apply_chat_template(olmocr_processor, olmocr_config, messages) # Pass messages list

    # Generate text
    page_generated_text = "" # Store generated text for current page
    try:
        # Get tokenizer reference for easier use
        tokenizer = olmocr_processor.tokenizer

        # print("\nStarting token generation for this page...")
        # Generate text and iterate over the tokens as they're generated
        for tokens in generate(
            olmocr_model,
            olmocr_processor,
            text_prompt,
            main_image,
            max_tokens = args.max_tokens,
            temperature = args.temperature,
        ):

            # Handle different token types (using modified logic from Action 11b)
            chunk = ""
            if isinstance(tokens, str):
                chunk = tokens
            elif hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
                if not all(isinstance(t, int) for t in tokens):
                    tokens = [int(t) for t in tokens if str(t).strip()]
                chunk = tokenizer.decode(tokens, skip_special_tokens=True)

            if not chunk:
                continue

            page_generated_text += chunk

    except Exception as e:
        print(f"\nError during generation for page {page_num}: {e}")
        import traceback
        traceback.print_exc()

    # --- Create PageResult object (like in pipeline.py) ---
    # No need to import PageResult again, already imported on line 20

    # Calculate char_span before creating PageResult
    start_pos = current_char_pos  # Track start position
    document_text += page_generated_text.strip() + ("\n" if page_num < num_pages else "") # Accumulate text
    current_char_pos = len(document_text) # Update current character position

    # Create PageResult instance without char_span
    page_result = PageResult(
        s3_path=pdf_filepath,  # Use pdf_filepath as s3_path for now
        page_num=page_num,
        response=PageResponse(natural_text=page_generated_text.strip(), primary_language=args.language, 
            is_rotation_valid=True, rotation_correction=0, is_table=False, is_diagram=False),
        input_tokens=0,  # Dummy values for now
        output_tokens=0,  # Dummy values for now
        is_fallback=False,  # Not fallback for now
    )
    page_results.append(page_result)
    char_spans.append([start_pos, current_char_pos])  # Store char span separately


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
# print(dolma_doc) # Print the Dolma document JSON

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
