from fastapi import FastAPI
import torch
from diffusers import ZImagePipeline
import io
import base64
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    prompt: str = "یک دختر ایرانی زیبا با لباس سنتی در منظره کوهستانی، غروب آفتاب"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0
    seed: int | None = None

# لود مدل یک بار موقع استارت
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
print("مدل Z-Image-Turbo با موفقیت لود شد!")

@app.post("/")
def generate(input: Input):
    generator = None
    if input.seed is not None:
        generator = torch.Generator("cuda").manual_seed(input.seed)

    image = pipe(
        prompt=input.prompt,
        height=input.height,
        width=input.width,
        num_inference_steps=input.num_inference_steps,
        guidance_scale=input.guidance_scale,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "prompt": input.prompt,
        "image_base64": image_base64
    }
