import ollama
import json
import re

class WhiteAgent:
    """
    A General Purpose Agent (Approach II).
    It receives a task (message + image) and responds.
    It does not know about the benchmark, only the context provided in the message.
    """
    def __init__(self, model_name="moondream"):
        self.model_name = model_name

    def _clean_json(self, text):
        """Generic JSON cleanup utility."""
        try:
            return json.loads(text)
        except:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except: pass
            
            return {"raw_output": text}

    def receive_task(self, message, image_path=None):
        """
        Receives the Self-Explanatory Prompt from Green Agent.
        """
        messages = [{'role': 'user', 'content': message}]
        
        if image_path:
             messages[0]['images'] = [image_path]

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                format="json", 
                options={'temperature': 0.6}
            )
            
            raw_content = response['message']['content']
            return self._clean_json(raw_content)
            
        except Exception as e:
            return {"error": str(e)}