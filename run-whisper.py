#from langchain_community.llms import Ollama
# import gradio as gr
import json
import os
import whisper
import torch
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Run Whisper speech-to-text on a recording.")
parser.add_argument("--model_size", type=str, default="base", help="Model size")
parser.add_argument("rec_filepath", help="Path of recording file to read")

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_size = args.model_size

def transcribe_audio(audio_file):
    model = whisper.load_model(model_size, device=DEVICE)
    audio = whisper.load_audio(audio_file,sr=16000)
    audio_tensor = torch.from_numpy(audio).to(torch.float32)
    result = model.transcribe(audio_tensor, fp16=False) #['text']
    return result

print(f"Model size: {model_size}")
# transcript = transcribe_audio("/Users/pbinkley/Documents/Projects/llm/whisper-mps/ClaudeRoberto/audio1630619881.m4a")


print(os.listdir())

source_file = args.rec_filepath
print(source_file)
transcript = transcribe_audio(source_file)

for seg in transcript['segments']:
	print(seg['text']) # note: segments have carriage return 
