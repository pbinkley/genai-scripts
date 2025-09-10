# genai-scripts

Some python scripts to run a couple of OCR or audio-transcript GenAI tools

## Steps

1. Create a Python virtual environment:

> ```python -m venv venv```

2. Activate the virtual environment (you will have to do this every time you start a new session):

> ```source venv/bin/activate```

3. Install the necessary python libraries (this will download about 1.5GB of python libraries and store them in the venv directory):

> ```pip install -r requirements.txt```

4. It is ready to go. Run the run-olmocr script on an image pdf:

> ```python run-olmocr.py my-test.pdf```

The output file will be written to a file ```my-test.txt``` in a new subdirectory called ```output```.
