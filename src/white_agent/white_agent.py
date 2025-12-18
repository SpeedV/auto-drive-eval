import ollama
import json
import re

class WhiteAgent:
    """
    Standard Approach 2 Agent.
    """
    def __init__(self, model_name="moondream"):
        self.model_name = model_name

    def _clean_json(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try: return json.loads(match.group())
                except: pass
            return {"raw_output": text}

    def receive_task(self, message, image_path=None):
        # --- FIX: NARRATIVE PROMPT ---
        # Forces detailed natural language descriptions, not data fields.
        system_prompt = (
            "ROLE: Autonomous Vehicle AI.\n"
            "TASK: Analyze the image and provide a driving log.\n"
            "REQUIREMENTS:\n"
            "1. Perception: Describe hazards, signs, and lights in detail.\n"
            "2. Prediction: Predict movement of cars/pedestrians.\n"
            "3. Planning: State a decisive action (e.g., 'Stop', 'Yield', 'Steer Left').\n"
            "   - Do NOT say 'might' or 'if'. Be specific.\n"
            "\n"
            f"{message}"
        )

        messages = [{'role': 'user', 'content': system_prompt}]
        
        if image_path:
             messages[0]['images'] = [image_path]

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                format="json", 
                options={'temperature': 0} 
            )
            
            raw_content = response['message']['content']
            return self._clean_json(raw_content)
            
        except Exception as e:
            return {"error": str(e)}