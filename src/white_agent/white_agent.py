import os
import json
import re
import base64
from openai import OpenAI

class WhiteAgent:
    """
    AutoDrive Agent (OpenAI Version).
    """
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        # Ensure OPENAI_API_KEY is set in environment variables
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _encode_image(self, image_path):
        """Encodes local image to base64 for OpenAI."""
        if not image_path or not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def _clean_json(self, text):
        try:
            return json.loads(text)
        except:
            # Strip markdown code blocks if present
            text = re.sub(r'```json', '', text)
            text = re.sub(r'```', '', text)
            try: return json.loads(text.strip())
            except: return {"raw_output": text}

    def receive_task(self, message, image_path=None):
        # Narrative Prompt to force detailed driving logic
        system_prompt = (
            "ROLE: Autonomous Vehicle AI.\n"
            "TASK: Analyze the image and provide a driving log.\n"
            "REQUIREMENTS:\n"
            "1. Perception: Describe hazards, signs, and lights in detail.\n"
            "2. Prediction: Predict movement of cars/pedestrians.\n"
            "3. Planning: State a decisive action (e.g., 'Stop', 'Yield', 'Steer Left').\n"
            "   - Do NOT say 'might' or 'if'. Be specific.\n"
            "\n"
            f"INSTRUCTION: {message}"
        )

        content_payload = [{"type": "text", "text": system_prompt}]
        
        # Attach image if provided
        if image_path:
            b64_img = self._encode_image(image_path)
            if b64_img:
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content_payload}],
                temperature=0,
                response_format={"type": "json_object"} # Force valid JSON
            )
            return self._clean_json(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}