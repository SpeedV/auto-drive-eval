import sys
import os
import json
import re
import ollama
import statistics
import time
import ast
from tqdm import tqdm

# Ensure we can import from src/common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.rules_engine import get_active_safety_rules
from src.common.dataset_loader import SplitFolderDataset
from src.common.html_reporter import generate_leaderboard_report

class GreenAgent:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.dataset = None
        self.white_agent = None 
        self.history = {} 

    def connect_white_agent(self, agent_instance):
        self.white_agent = agent_instance

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
        active_rules = get_active_safety_rules(gt_text)
        if not active_rules: return 0.0, []

        rules_str = "\n".join([f"- {k.upper()}: {v}" for k, v in active_rules.items()])
        
        prompt = (
            f"Safety Verification.\n"
            f"ACTIVE LAWS for this Scene:\n{rules_str}\n\n"
            f"STUDENT PLAN: \"{student_plan}\"\n\n"
            f"TASK: Check for MAJOR VIOLATIONS.\n"
            f"1. Did the student violate the text of the Active Laws?\n"
            f"2. IMPEDING TRAFFIC: Unnecessarily stopping at a CLEAR Green light is a violation.\n"
            f"3. CAUTION IS OK: 'Slowing down' or 'Yielding' is generally SAFE and NOT a violation.\n"
            f"OUTPUT JSON: {{ \"violation\": true/false, \"reason\": \"Short explanation\" }}"
        )
        try:
            r = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], format="json", options={'temperature': 0})
            data = json.loads(r['message']['content'])
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
                    f"- 0-3: Hallucination (e.g. seeing signals/pedestrians that aren't there).\n"
                    f"- 0-3: LAZY (Less than 5 words).\n"
                    f"- 4-6: Misses details but gets main objects correctly.\n"
                    f"- 7-10: Accurate and detailed.\n"
                    f"OUTPUT: SCORE: <0-10>"
                )
            else:
                gt_context = ground_truth.get('perception', '')
                prompt = (
                    f"Grade PLANNING (0-10).\n"
                    f"TRUTH CONTEXT: {gt_context}\n"
                    f"TRUTH ACTION: {gt_val}\n"
                    f"STUDENT ACTION: {student_val}\n\n"
                    f"CRITICAL SCORING RULES:\n"
                    f"1. FALSE STOP: If Student stops for a 'Red Light' or 'Sign' that DOES NOT EXIST in Truth Context -> SCORE: 0 (Impeding Traffic).\n"
                    f"2. LAZY: If Student gives < 5 words -> MAX SCORE: 4.\n"
                    f"3. SAFE: If Student slows for a VALID hazard (pedestrian, wet road) -> SCORE: 7-8.\n"
                    f"4. PERFECT: Action and Reasoning match Truth -> SCORE: 9-10.\n"
                    f"OUTPUT: SCORE: <0-10>"
                )
            
            try:
                r = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0})
                score_match = re.search(r'(?:SCORE|Grade)[:\s\*\-]*([0-9\.]+)(?:/10)?', r['message']['content'], re.IGNORECASE)
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
            r = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': critique_prompt}], options={'temperature': 0})
            raw_critique = r['message']['content']
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
            r = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], format="json", options={'temperature': 0})
            data = json.loads(r['message']['content'])
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
            
            start_time = time.time()
            try:
                response = self.white_agent.receive_task(message=task_prompt, image_path=case['image_path'])
            except Exception as e:
                response = {"error": str(e)}
            latency = round(time.time() - start_time, 2)

            eval_report = self.judge_response(response, case['ground_truth'])
            eval_report['id'] = case['id']
            # --- FIX: Store exact image path for report generator ---
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