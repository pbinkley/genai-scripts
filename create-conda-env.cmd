call conda deactivate
call conda create --name olmocr python=3.11 --yes
call conda activate olmocr
call conda install Pillow PyPDF2 validators jiwer --yes

# CUDA 12.8 is not available from conda, so we must use pip
# we use "python -m pip" because in Windows we can't use "yes | pip"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128
python test.py

