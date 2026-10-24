

import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="AI Creator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    style: Literal["image", "cartoon"] = "image"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/image")
def generate_image(body: GenerateRequest):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Backend API key is not configured."
        )

    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    prompt = body.prompt.strip()

    if body.style == "cartoon":
        prompt = (
            "Create a polished cartoon illustration with expressive "
            "characters, appealing colors, clean shapes, and no watermark. "
            + prompt
        )

    try:
        client = OpenAI(api_key=api_key)
        result = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
        )

        image_data = result.data[0].b64_json

        if not image_data:
            raise HTTPException(
                status_code=502,
                detail="No image was returned by the AI provider."
            )

        return {
            "mime_type": "image/png",
            "image_base64": image_data,
            "model": model,
        }

    except HTTPException:
        raise
    except OpenAIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Image generation failed: {type(exc).__name__}"
        ) from exc


@app.post("/api/video")
def generate_video():
    raise HTTPException(
        status_code=501,
        detail="Video generation integration is not configured yet."
    )
