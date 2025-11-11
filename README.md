# LookingGlass-RGBD-Holo-Image-
Converts any image to RGBD for Looking Glass Portrait using Distill Any Depth

Interactive depth-estimation demo based on the Hugging Face Space  
[https://huggingface.co/spaces/xingyang1/Distill-Any-Depth](https://huggingface.co/spaces/xingyang1/Distill-Any-Depth).

## What it does
Upload an image → get  
- coloured depth map (slider)  
- grey-scale depth PNG  
- raw NumPy depth array  
- side-by-side RGBD PNG for use in Looking Glass Portrait(`original + grey-depth`)

## 1. Install (Windows 11 + Python 3.11 shown)

# 1. use Python 3.10 / 3.11 (3.12+ lacks wheels for some deps)
py -3.11 -m venv venv
.\venv\Scripts\activate

# 2. upgrade pip
python -m pip install -U pip setuptools wheel

# 3. install **pinned** versions that work together
python -m pip install -r requirements.txt

# 4. fix the HfFolder / JSON-schema bug that ships with 4.36
python -m pip install --upgrade "gradio-client&gt;=0.16.2"
