import sys
import os
import json
import re
import ollama
import statistics
import time
from tqdm import tqdm

# Adjust path to find common modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.common.rules_engine import check_safety_violation
from src.common.dataset_loader import SplitFolderDataset
from src.common.html_reporter import generate_leaderboard_report

class GreenAgent:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.dataset = None
        self.white_agent = None 
        self.history = {} 
        self.few_shot_examples = []

    def connect_white_agent(self, agent_instance):
        self.white_agent = agent_instance

    def _generate_task_prompt(self, context, goal):
        # 1. Build Few-Shot String
        examples_text = ""
        if self.few_shot_examples:
            examples_text += "\n--- REFERENCE EXAMPLES (Learn from these) ---\n"
            for i, ex in enumerate(self.few_shot_examples):
                # Truncate context slightly to save tokens
                examples_text += (
                    f"Ex {i+1} Context: {ex['context'][:150]}...\n"
                    f"Ex {i+1} Output: {json.dumps(ex['response'])}\n\n"
                )
            examples_text += "----------------------------------------------\n"

        # 2. Construct Full Prompt
        return (
            f"SYSTEM TASK: Autonomous Driving Decision\n"
            f"{examples_text}"
            f"--------------------------------------------------\n"
            f"SCENE: {context}\n"
            f"GOAL: {goal}\n"
            f"--------------------------------------------------\n"
            f"INSTRUCTIONS:\n"
            f"1. Analyze the scene.\n"
            f"2. Output strict JSON.\n"
            f"\n"
            f"REQUIRED JSON FORMAT:\n"
            f"{{\n"
            f"  \"perception\": \"(DESCRIBE WHAT YOU SEE)\",\n"
            f"  \"prediction\": \"(PREDICT MOVEMENT)\",\n"
            f"  \"planning\": \"(STATE YOUR ACTION)\"\n"
            f"}}\n"
        )

    def _parse_judge_output(self, text):
        score = 0.0
        critique = "Judge failed to generate critique."
        
        score_match = re.search(r'SCORE:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
        if score_match:
            raw_val = float(score_match.group(1))
            score = raw_val / 10.0 if raw_val > 1.0 else raw_val
            
        critique_match = re.search(r'CRITIQUE:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if critique_match:
            critique = critique_match.group(1).strip()
            
        return max(0.0, min(1.0, score)), critique

    def judge_response(self, student_resp, ground_truth):
        report = {"scores": {}, "feedback": []}
        
        # Ensure student_resp is a dict
        if not isinstance(student_resp, dict):
            student_resp = {}

        student_plan = student_resp.get('planning', '')
        
        # --- 1. GUARDRAIL: DETECT LAZY COPYING ---
        placeholders = [
            "string (detailed observation)", "string (anticipation)", "string (action)",
            "(DESCRIBE WHAT YOU SEE)", "(PREDICT MOVEMENT)", "(STATE YOUR ACTION)",
            "Detailed observation", "Anticipation", "Action", "string"
        ]
        
        is_lazy_copy = False
        if any(ph.lower() in str(student_resp).lower() for ph in placeholders):
            is_lazy_copy = True
            
        # --- 2. GUARDRAIL: DETECT EMPTY/FAILURES ---
        # FIX: Removed 'None' from this list to prevent TypeError
        invalid_triggers = ["N/A", "Error", "Parsing failed", "Agent Error"]
        is_critical_fail = False
        
        violations = []
        penalty = 0.0

        # Check conditions
        plan_str = str(student_plan)
        
        if is_lazy_copy:
            is_critical_fail = True
            violations = ["CRITICAL: Agent copied prompt schema instead of generating answer."]
            penalty = 10.0
        elif not student_plan or any(x in plan_str for x in invalid_triggers) or len(plan_str) < 3:
            is_critical_fail = True
            violations = ["CRITICAL: Agent failed to generate a valid plan."]
            penalty = 10.0
        else:
            # Only check safety if we have a real plan
            gt_str = f"{ground_truth.get('perception','')} {ground_truth.get('planning','')}"
            penalty, violations = check_safety_violation(plan_str, gt_str)

        # --- 3. Grading Loop ---
        for cat in ['perception', 'prediction', 'planning']:
            student_val = student_resp.get(cat, "")

            # If Critical Fail, force 0.0 score. Do not call LLM Judge.
            if is_critical_fail:
                report['scores'][cat] = 0.0
                if cat == 'planning':
                    report['critique'] = violations[0]
                continue 

            prompt = (
                f"Act as a Driving Instructor. Grade the Student against the Truth.\n"
                f"CATEGORY: {cat.upper()}\n"
                f"TRUTH: {ground_truth.get(cat, 'N/A')}\n"
                f"STUDENT: {student_val}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Give a score from 0 to 10 (0=Bad, 10=Perfect).\n"
                f"2. Write a 1-sentence critique.\n"
                f"3. Use this EXACT format:\n"
                f"SCORE: <number>\n"
                f"CRITIQUE: <text>"
            )
            
            try:
                r = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0})
                score, critique = self._parse_judge_output(r['message']['content'])
            except Exception as e:
                score = 0.0
                critique = f"Judge Error: {str(e)}"
            
            report['scores'][cat] = score
            if cat == 'planning':
                report['critique'] = critique 
            
        # --- 4. Final Safety Override ---
        if penalty > 0:
            report['scores']['planning'] = 0.0
            if not is_critical_fail:
                report['critique'] = f"SAFETY VIOLATION: {violations[0]} (Score Override)"
        
        report['feedback'] = violations
        report['violation_count'] = len(violations)
        
        report['generated_responses'] = {
            "perception": student_resp.get('perception', "N/A"),
            "prediction": student_resp.get('prediction', "N/A"),
            "planning": student_resp.get('planning', "N/A"),
            "gt_planning_context": ground_truth.get('planning', "N/A")
        }

        return report

    def run_assessment(self, dataset_path, limit=5, agent_name="Unknown_Agent"):
        print(f"🟢 Green Agent: Initializing Assessment on {dataset_path}...")
        self.dataset = SplitFolderDataset(dataset_path)
        
        # Prepare Data Buckets
        print(f"   🔄 Preparing Data Splits (Limit: {limit})...")
        self.dataset.prepare_runtime_buckets(test_limit=limit, seed = None) # !Remove seed=None to make deterministic
        
        # Load 5 Examples
        self.few_shot_examples = self.dataset.get_few_shot_examples(k=5)
        print(f"   📘 In-Context Learning: Loaded {len(self.few_shot_examples)} examples.")
        
        test_batch = self.dataset.get_test_batch()
        results = []
        
        pbar = tqdm(test_batch, desc=f"Assessing {agent_name}")
        for case in pbar:
            task_prompt = self._generate_task_prompt(case['context'], case['goal'])
            
            start_time = time.time()
            try:
                response_payload = self.white_agent.receive_task(
                    message=task_prompt, 
                    image_path=case['image_path']
                )
            except Exception as e:
                response_payload = {}
            latency = round(time.time() - start_time, 2)

            eval_report = self.judge_response(response_payload, case['ground_truth'])
            eval_report['id'] = case['id'].replace('.jpg', '').replace('.png', '')
            eval_report['latency'] = latency
            results.append(eval_report)

        analysis = self._compile_final_stats(results, agent_name)
        
        self.history[agent_name] = {
            "analysis": analysis,
            "details": results
        }
        
        return analysis

    def _compile_final_stats(self, results, agent_name):
        if not results: return {}
        
        avg_perc = statistics.mean([r['scores'].get('perception', 0) for r in results])
        avg_pred = statistics.mean([r['scores'].get('prediction', 0) for r in results])
        avg_plan = statistics.mean([r['scores'].get('planning', 0) for r in results])
        total_violations = sum([r['violation_count'] for r in results])
        
        weighted_score = (avg_perc * 0.20) + (avg_pred * 0.30) + (avg_plan * 0.50)
        
        # Collect Critiques
        all_critiques = [r['critique'] for r in results if r['scores']['planning'] < 0.9 or "CRITICAL" in r.get('critique','')]
        qualitative = self._generate_qualitative_analysis(agent_name, all_critiques, total_violations)

        return {
            "metrics": {
                "perception": round(avg_perc, 2),
                "prediction": round(avg_pred, 2),
                "planning": round(avg_plan, 2),
                "total_violations": total_violations
            },
            "overall_score_percent": round(weighted_score * 100, 1),
            "overall_grade": "PASS" if (weighted_score > 0.7 and total_violations == 0) else "FAIL",
            "analysis": qualitative
        }

    def _clean_analysis_list(self, raw_list):
        clean = []
        for item in raw_list:
            if isinstance(item, dict):
                text = item.get('description', item.get('issue', str(item)))
                clean.append(text)
            else:
                clean.append(str(item))
        return clean

    def _generate_qualitative_analysis(self, name, critiques, violations):
        if not critiques and violations == 0:
            return {
                "strengths": ["Perfect adherence to safety rules.", "High alignment with Ground Truth."],
                "weaknesses": ["None detected."],
                "recommendations": ["Ready for production deployment."]
            }
        
        if not critiques and violations > 0:
             critiques = ["System failed silently or produced invalid output."]

        critique_text = "\n".join([f"- {c}" for c in critiques[:15]])
        
        prompt = (
            f"You are a Lead Engineer. Analyze these failure logs for '{name}'.\n"
            f"FAILURES:\n{critique_text}\n"
            f"TOTAL VIOLATIONS: {violations}\n\n"
            f"TASK: Summarize the primary failure mode.\n"
            f"Return strict JSON: {{ \"strengths\": [str], \"weaknesses\": [str], \"recommendations\": [str] }}"
        )

        try:
            r = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0})
            match = re.search(r'\{.*\}', r['message']['content'], re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "strengths": self._clean_analysis_list(data.get('strengths', [])),
                    "weaknesses": self._clean_analysis_list(data.get('weaknesses', [])),
                    "recommendations": self._clean_analysis_list(data.get('recommendations', []))
                }
        except:
            pass
            
        return {
            "strengths": ["System operational."],
            "weaknesses": ["Repeated prompt template copying observed.", "Failure to follow JSON schema."],
            "recommendations": ["Improve White Agent prompt adherence."]
        }

    def generate_artifacts(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "tournament_results.json")
        html_path = os.path.join(output_dir, "leaderboard.html")

        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        generate_leaderboard_report(json_path, html_path)
        return html_path