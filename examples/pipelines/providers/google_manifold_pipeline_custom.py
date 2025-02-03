from typing import List, Union, Iterator
import os

from pydantic import BaseModel, Field

import google.generativeai as genai
from google.generativeai.types import GenerationConfig


class Pipeline:
    """Google GenAI pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_API_KEY: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)
        ALLOWED_MODELS: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "google_genai"
        self.name = "Google: "

        self.valves = self.Valves(**{
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            "USE_PERMISSIVE_SAFETY": False,
            "ALLOWED_MODELS": os.getenv("ALLOWED_MODELS", "")
        })
        self.pipelines = []

        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_startup(self) -> None:
        print(f"on_startup:{__name__}")
        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_shutdown(self) -> None:
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        print(f"on_valves_updated:{__name__}")
        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    def update_pipelines(self) -> None:
        """Update the available models from Google GenAI"""

        if self.valves.GOOGLE_API_KEY:
            try:
                models = genai.list_models()
                allowed_models = [m.strip() for m in self.valves.ALLOWED_MODELS.split(",") if m.strip()]

                self.pipelines = [
                    {
                        "id": model.name[7:],
                        "name": model.display_name,
                    }
                    for model in models
                    if "generateContent" in model.supported_generation_methods
                    if model.name[:7] == "models/"
                    if not allowed_models or model.display_name in allowed_models
                ]
            except Exception:
                self.pipelines = [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Google, please update the API Key in the valves.",
                    }
                ]
        else:
            self.pipelines = []

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Iterator]:
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"

        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)

            if model_id.startswith("google_genai."):
                model_id = model_id[12:]
            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)

            contents = []
            for message in messages:
                if message["role"] != "system":
                    if isinstance(message.get("content"), list):
                        parts = []
                        for content in message["content"]:
                            if content["type"] == "text":
                                parts.append({"text": content["text"]})
                            elif content["type"] == "image_url":
                                image_url = content["image_url"]["url"]
                                if image_url.startswith("data:image"):
                                    image_data = image_url.split(",")[1]
                                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_data}})
                                else:
                                    parts.append({"image_url": image_url})
                        contents.append({"role": message["role"], "parts": parts})
                    else:
                        contents.append({
                            "role": "user" if message["role"] == "user" else "model",
                            "parts": [{"text": message["content"]}]
                        })

            model = genai.GenerativeModel(model_name=model_id, system_instruction=system_message)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            } if self.valves.USE_PERMISSIVE_SAFETY else body.get("safety_settings")

            response = model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=body.get("stream", False),
            )

            return self.stream_response(response) if body.get("stream", False) else response.text
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                yield chunk.text
