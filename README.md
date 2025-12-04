# genai-scripts

Some python scripts to run a couple of OCR or audio-transcript GenAI tools

## Steps for run-olmocr.py to perform OCR on an image pdf

1. Create a Python virtual environment to hold the dependencies, making it easy to delete all this when you don't need it any more.

1a. ... using [native Python venv](https://docs.python.org/3/library/venv.html):

> ```python -m venv venv``` # creates local directory ```venv```

1b. ... or using [Anaconda](https://www.anaconda.com/docs/getting-started/working-with-conda/environments):

> ```conda create -n genai-scripts anaconda``` # creates a virtual environment named ```genai-scripts``` in Anaconda's 

2. Activate the virtual environment (**you will have to do this every time you start a new session**):

> ```source venv/bin/activate```

... or ...

> ```conda activate genai-scripts```

3. Install the necessary python libraries (this will download about 1.5GB of python libraries and store them in the venv directory or the Anaconda virtual environment):

> ```pip install -r requirements.txt```

4. It is ready to go. Run the run-olmocr script on an image pdf:

> ```python run-olmocr.py my-test.pdf```

The first time you run this, Ollama will download and install the olmocr model. The output file will be written to a file ```my-test.txt``` in a new subdirectory called ```output```.

5. Delete the virtual environment when you don't need it any more
