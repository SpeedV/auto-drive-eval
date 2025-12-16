import ollama
import json
import re

class WhiteAgent:
    def __init__(self, model_name):
        self.variant = "standard"
        if "moondream-cautious" in model_name:
            self.model = "moondream"
            self.variant = "cautious"
        elif "moondream-aggressive" in model_name:
            self.model = "moondream"
            self.variant = "aggressive"
        else:
            self.model = model_name
            self.variant = "standard"
        self.training_context = ""

    def train(self, training_examples):
        if not training_examples: return
        self.training_context = "EXAMPLES:\n"
        for ex in training_examples:
            self.training_context += (
                f"- Context: {ex.get('context')}\n"
                f"  Plan: \"{ex.get('planning')}\"\n"
            )

    def _get_system_prompt(self):
        base = "You are an autonomous driving AI."
        if self.variant == "cautious":
            return f"{base} You are EXTREMELY CAUTIOUS. Focus on risks."
        elif self.variant == "aggressive":
            return f"{base} You are EFFICIENT. Focus on speed."
        return f"{base} Provide clear driving reasoning."

    def _clean_output(self, value, fallback=""):
        if not value: return fallback
        if isinstance(value, (list, tuple)): return " ".join(map(str, value))
        if isinstance(value, dict): return ". ".join([f"{k}: {v}" for k, v in value.items()])
        text = str(value).strip()
        # Remove markdown bolding if present
        text = text.replace("**", "").replace("*", "")
        return text

    def _robust_json_parse(self, text):
        """
        Attempts to parse JSON. If standard parsing fails (BakLLaVA issue),
        uses Regex to extract keys manually.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: Regex extraction for broken JSON
            data = {}
            
            # Pattern to find "key": "value" (non-greedy)
            # We handle both single and double quotes
            patterns = {
                "perception": r'[\"\']perception[\"\']\s*:\s*[\"\'](.*?)[\"\'](?:,|}|\n)',
                "prediction": r'[\"\']prediction[\"\']\s*:\s*[\"\'](.*?)[\"\'](?:,|}|\n)',
                "planning": r'[\"\']planning[\"\']\s*:\s*[\"\'](.*?)[\"\'](?:,|}|\n)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    data[key] = match.group(1)
                else:
                    data[key] = f"Error parsing {key} from raw output."
            
            return data

    def generate_response(self, image_path, context, goal):
        system_instruction = self._get_system_prompt()
        
        if "moondream" in self.model:
            prompt = (
                f"{system_instruction}\n"
                f"TASK: Analyze image. Context: {context}. Goal: {goal}.\n"
                "OUTPUT: A JSON object with 3 keys: 'perception', 'prediction', 'planning'.\n"
                "VALUES: Must be simple sentences describing the scene and your action.\n"
                "EXAMPLE: {\"perception\": \"I see a red car.\", \"prediction\": \"It will stop.\", \"planning\": \"I will brake.\"}"
            )
        else:
            prompt = (
                f"{system_instruction}\n"
                f"{self.training_context}\n"
                f"SCENE CONTEXT: {context}\n"
                f"GOAL: {goal}\n\n"
                "TASK: Analyze the image. Return a JSON object with keys 'perception', 'prediction', 'planning'.\n"
                "IMPORTANT: The values must be NATURAL LANGUAGE PARAGRAPHS. Do NOT use lists.\n"
                "If you see nothing, say 'I see clear road'. Do not leave blank."
            )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
                format="json",
                options={'num_ctx': 4096, 'temperature': 0.6}
            )
            raw_content = response['message']['content']
            
            data = self._robust_json_parse(raw_content)
            
            perc = self._clean_output(data.get("perception"), "No perception generated.")
            pred = self._clean_output(data.get("prediction"), "No prediction generated.")
            plan = self._clean_output(data.get("planning"), "No plan generated.")

            # Lazy Output Check
            if len(perc) < 3 or perc.lower() == "obstacle":
                perc = f"The agent detected an obstacle but failed to describe it in detail. (Raw: {perc})"

            return {"perception": perc, "prediction": pred, "planning": plan}

        except Exception as e:
            return {
                "perception": "Error generating perception.", 
                "prediction": "Error generating prediction.", 
                "planning": f"Error: {str(e)}"
            }