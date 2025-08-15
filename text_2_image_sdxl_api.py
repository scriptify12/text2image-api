### text2image_sdxl_api

# --- main.py ---

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline
import torch
import uuid
import os
from PIL import Image
from fastapi.responses import FileResponse


app = FastAPI()

# Load model at startup
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
hf_token = os.getenv("HUGGINGFACE_TOKEN")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    token=hf_token
).to("cuda")
pipe.enable_attention_slicing()

# Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request schema
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_image(data: PromptRequest):
    try:
        image = pipe(data.prompt).images[0]
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        image.save(path)
        return {"success": True, "filename": filename, "url": f"/images/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{filename}")
def get_image(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")


# --- requirements.txt ---

fastapi
uvicorn
torch
transformers
diffusers[torch]>=0.22.0
Pillow


# --- Dockerfile ---

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y git python3 python3-pip

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENV HUGGINGFACE_TOKEN=your_token_here

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]


# --- README.md ---

## Text-to-Image API with SDXL + FastAPI

### Instructions

1. Replace `your_token_here` in `Dockerfile` or set `HUGGINGFACE_TOKEN` as an env var.
2. Run locally with GPU or deploy to RunPod.

### Run Locally (GPU required)
```bash
pip install -r requirements.txt
HUGGINGFACE_TOKEN=your_token_here uvicorn main:app --reload --port 7860
```

### Run in Docker
```bash
docker build -t text2image .
docker run -e HUGGINGFACE_TOKEN=your_token_here -p 7860:7860 text2image
```

### API Endpoint
`POST /generate`
```json
{
  "prompt": "a cat wearing a wizard hat, fantasy art"
}
```
Returns:
```json
{
  "success": true,
  "filename": "...",
  "url": "/images/..."
}
```

### Notes
- Generated images are saved in `outputs/` folder
- Can be hosted on RunPod with GPU
