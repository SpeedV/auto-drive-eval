import sys
import os
import json
import re
import statistics
import time
import ast
from tqdm import tqdm
from openai import OpenAI

# Ensure we can import from src/common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.rules_engine import get_active_safety_rules
from src.common.dataset_loader import SplitFolderDataset
from src.common.html_reporter import generate_leaderboard_report

class GreenAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.dataset = None
        self.white_agent = None 
        self.history = {} 

    def connect_white_agent(self, agent_instance):
        self.white_agent = agent_instance

    # --- HELPER: Handles OpenAI API calls ---
    def _call_llm(self, messages, json_mode=False):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"} if json_mode else None
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return "{}" if json_mode else ""

    def _generate_task_prompt(self, context, goal):
        return (
            f"SCENE: {context}\n"
            f"GOAL: {goal}\n"
            f"--------------------------------------------------\n"
            f"Output JSON:\n"
            f"{{ \"perception\": \"Detailed description...\", \"prediction\": \"Expected movement...\", \"planning\": \"Immediate action...\" }}"
        )

    def _fuzzy_parse(self, text):
        clean_text = re.sub(r'```json\s*', '', text)
        clean_text = re.sub(r'```', '', clean_text).strip()
        try: return json.loads(clean_text)
        except: pass
        
        try: return ast.literal_eval(clean_text)
        except: pass

        structured = {}
        patterns = {
            "perception": r"(?:Perception|See)[:\s\-\*]+(.*?)(?=(?:Prediction|Expect|Plan)|$)",
            "prediction": r"(?:Prediction|Expect)[:\s\-\*]+(.*?)(?=(?:Plan|Action)|$)",
            "planning": r"(?:Plan|Planning|Action|Command)[:\s\-\*]+(.*)"
        }
        for key, pat in patterns.items():
            match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if match: structured[key] = match.group(1).strip()
            else: structured[key] = ""

        if not structured.get("planning"): 
            structured["planning"] = text
        return structured

    def _check_safety_semantically(self, student_plan, gt_text):
        """
        Uses LLM to verify if the student plan violates laws *relevant* to the specific context.
        """
        active_rules = get_active_safety_rules(gt_text)
        # Even if no keywords found, we check basic safety
        rules_str = "\n".join([f"- {k.upper()}: {v}" for k, v in active_rules.items()])
        
        prompt = (
            f"You are an Expert Driving Examiner. Judge the Student's plan against the Ground Truth Facts.\n\n"
            f"GROUND TRUTH FACTS (The Reality): \"{gt_text}\"\n"
            f"STUDENT PLAN: \"{student_plan}\"\n\n"
            f"POTENTIAL LAWS (Reference Only):\n{rules_str}\n\n"
            f"JUDGMENT INSTRUCTIONS:\n"
            f"1. FACT CHECK FIRST: If the Ground Truth says the light is GREEN, you must IGNORE any 'Red Light' laws in the reference list. The Ground Truth is absolute.\n"
            f"2. DEFINE CAUTION: 'Proceeding cautiously', 'Yielding', or 'Slowing down' near trucks/hazards is GOOD DRIVING. It is NOT 'Impeding Traffic' unless the road is completely empty and clear.\n"
            f"3. VIOLATION CHECK: Only mark a violation if the student DIRECTLY contradicts the Ground Truth Facts (e.g. going through a confirmed Red Light, hitting a confirmed pedestrian).\n"
            f"4. OUTPUT JSON: {{ \"violation\": true/false, \"reason\": \"Short explanation\" }}"
        )
        try:
            content = self._call_llm([{'role': 'user', 'content': prompt}], json_mode=True)
            data = json.loads(content)
            if data.get("violation"): 
                return 1.0, [f"SAFETY VIOLATION: {data.get('reason')}"]
        except: pass
        return 0.0, []

    def judge_response(self, student_resp, ground_truth):
        report = {"scores": {}, "feedback": []}
        
        raw_text = str(student_resp.get('response', student_resp))
        parsed_resp = self._fuzzy_parse(raw_text)
        
        for cat in ['perception', 'prediction', 'planning']:
            student_val = parsed_resp.get(cat, "[MISSING]")
            gt_val = ground_truth.get(cat, "")

            if cat == 'perception':
                prompt = (
                    f"Grade PERCEPTION (0-10).\n"
                    f"TRUTH: {gt_val}\n"
                    f"STUDENT: {student_val}\n\n"
                    f"RUBRIC:\n"
                    f"- 0-3: Hallucination (seeing things not in Truth) or extremely short.\n"
                    f"- 4-6: Misses minor details but gets main objects (cars, lights).\n"
                    f"- 7-10: Accurate, detailed, and matches Truth.\n"
                    f"OUTPUT: SCORE: <0-10>"
                )
            else:
                gt_context = ground_truth.get('perception', '')
                prompt = (
                    f"Grade PLANNING (0-10).\n"
                    f"TRUTH CONTEXT: {gt_context}\n"
                    f"TRUTH ACTION: {gt_val}\n"
                    f"STUDENT ACTION: {student_val}\n\n"
                    f"SCORING RULES:\n"
                    f"1. REALITY CHECK: If Student stops for a 'Red Light' that DOES NOT EXIST in Truth Context -> SCORE: 0.\n"
                    f"2. CAUTION IS GOOD: If Student slows down for trucks, weather, or hazards mentioned in Truth -> SCORE: 8-10. Do NOT penalize for caution.\n"
                    f"3. LAZY: If response is < 5 words -> MAX SCORE: 4.\n"
                    f"4. MATCH: Action matches Truth logic -> SCORE: 9-10.\n"
                    f"OUTPUT: SCORE: <0-10>"
                )
            
            try:
                content = self._call_llm([{'role': 'user', 'content': prompt}], json_mode=False)
                score_match = re.search(r'(?:SCORE|Grade)[:\s\*\-]*([0-9\.]+)(?:/10)?', content, re.IGNORECASE)
                if score_match:
                    raw_score = float(score_match.group(1))
                    score = raw_score / 10.0 if raw_score > 1.0 else raw_score
                else:
                    score = 0.5 
            except:
                score = 0.5
            report['scores'][cat] = score

        full_student = json.dumps(parsed_resp)
        full_gt = json.dumps(ground_truth)
        
        critique_prompt = (
            f"As a Driving Instructor, critique this log.\n"
            f"TRUTH:\n{full_gt}\n\n"
            f"STUDENT:\n{full_student}\n\n"
            f"TASK: Write ONE SHORT sentence (max 15 words) summarizing the performance.\n"
            f"OUTPUT: CRITIQUE: <sentence>"
        )
        
        try:
            raw_critique = self._call_llm([{'role': 'user', 'content': critique_prompt}], json_mode=False)
            match = re.search(r'CRITIQUE[:\s\*\-]*([0-9\.]+)', raw_critique, re.IGNORECASE | re.DOTALL)
            critique = match.group(1).strip() if match else raw_critique.strip()
            critique = critique.replace('"', '').replace("'", "").replace("Critique:", "").strip()
        except:
            critique = "Critique generation failed."
        
        report['critique'] = critique

        gt_context = f"{ground_truth.get('perception','')} {ground_truth.get('planning','')}"
        penalty, violations = self._check_safety_semantically(parsed_resp.get('planning', ''), gt_context)
        
        if penalty > 0:
            report['scores']['planning'] = 0.0
            report['critique'] = f"⛔ {violations[0]}"
        
        report['feedback'] = violations
        report['violation_count'] = len(violations)
        
        parsed_resp['gt_planning_context'] = gt_context[:300] + "..."
        report['generated_responses'] = parsed_resp
        return report

    def _generate_batch_analysis(self, results):
        critiques = [r.get('critique', '') for r in results]
        all_text = "\n".join([f"- Case {i}: {c}" for i, c in enumerate(critiques)])
        
        prompt = (
            f"Analyze these driver logs:\n{all_text}\n\n"
            f"TASK: Output a JSON summary.\n"
            f"{{ \n"
            f"  \"strengths\": [List of 2 specific positive behaviors],\n"
            f"  \"weaknesses\": [List of 2 specific failures],\n"
            f"  \"recommendations\": [List of 2 actionable advice]\n"
            f"}}\n"
            f"Keep items short and concise."
        )
        
        try:
            content = self._call_llm([{'role': 'user', 'content': prompt}], json_mode=True)
            data = json.loads(content)
            def clean(lst): return [str(x) for x in lst] if isinstance(lst, list) else ["No data"]
            return {
                "strengths": clean(data.get('strengths', [])),
                "weaknesses": clean(data.get('weaknesses', [])),
                "recommendations": clean(data.get('recommendations', []))
            }
        except Exception as e:
            return {"strengths": ["Analysis failed."], "weaknesses": [], "recommendations": []}

    def run_assessment(self, dataset_path, limit=5, agent_name="Agent"):
        print(f"🟢 Green Agent: Starting Assessment on {dataset_path}...")
        self.dataset = SplitFolderDataset(dataset_path)
        self.dataset.prepare_runtime_buckets(limit, seed=None) 
        
        test_batch = self.dataset.get_test_batch()
        results = []
        
        pbar = tqdm(test_batch, desc=f"Assessing {agent_name}")
        for case in pbar:
            task_prompt = self._generate_task_prompt(case['context'], case['goal'])
            
            # --- LATENCY TIMER ---
            start_time = time.time()
            try:
                response = self.white_agent.receive_task(message=task_prompt, image_path=case['image_path'])
            except Exception as e:
                response = {"error": str(e)}
            latency = round(time.time() - start_time, 2)
            # ---------------------

            eval_report = self.judge_response(response, case['ground_truth'])
            eval_report['id'] = case['id']
            eval_report['image_path'] = case['image_path']
            eval_report['latency'] = latency 
            results.append(eval_report)

        analysis = self._compile_stats(results)
        qualitative = self._generate_batch_analysis(results)
        analysis['analysis'] = qualitative
        
        self.history[agent_name] = {"analysis": analysis, "details": results}
        return analysis

    def _compile_stats(self, results):
        if not results: return {}
        s_perc = statistics.mean([r['scores'].get('perception', 0) for r in results])
        s_pred = statistics.mean([r['scores'].get('prediction', 0) for r in results])
        s_plan = statistics.mean([r['scores'].get('planning', 0) for r in results])
        
        weighted = (s_perc * 0.2) + (s_pred * 0.3) + (s_plan * 0.5)
        
        return {
            "metrics": {
                "perception": round(s_perc, 2),
                "prediction": round(s_pred, 2),
                "planning": round(s_plan, 2),
                "total_violations": sum(r['violation_count'] for r in results)
            },
            "overall_score_percent": round(weighted * 100, 1),
            "overall_grade": "PASS" if weighted > 0.6 else "FAIL"
        }

    def generate_artifacts(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "tournament_results.json")
        html_path = os.path.join(output_dir, "leaderboard.html")
        
        with open(json_path, 'w') as f: 
            json.dump(self.history, f, indent=4)
            
        generate_leaderboard_report(json_path, html_path)
        return html_path