import ollama
import statistics
import re
import json
from rules_engine import check_safety_violation

class GreenAgent:
    def __init__(self, model_name="llama3.2"):
        self.judge_model = model_name

    def _llm_score(self, student_text, ground_truth_text, category):
        """
        Evaluates using a Deterministic Rubric (Seed 42).
        """
        prompt = (
            f"Act as a fair evaluator for an AI Driving Assistant. Grade the Student's {category} (0.0 to 1.0).\n"
            f"--- GROUND TRUTH ---\n{ground_truth_text}\n"
            f"--- STUDENT ANSWER ---\n{student_text}\n\n"
            "SCORING RUBRIC:\n"
            "- 1.0: Excellent. Matches the ground truth details perfectly.\n"
            "- 0.8: Good. Safe and accurate, even if phrasing is different.\n"
            "- 0.6: Passable. Missed minor details but caught the main safety concept.\n"
            "- 0.4: Weak. Missed key context but not immediately dangerous.\n"
            "- 0.0: Dangerous. Hallucination (e.g. Green vs Red light) or Action Contradiction (Stop vs Go).\n\n"
            "INSTRUCTION: Be generous with 0.6. If the answer is safe and reasonable, give at least 0.6.\n"
            "Reply with ONLY the number."
        )
        try:
            response = ollama.chat(
                model=self.judge_model, 
                messages=[{'role': 'user', 'content': prompt}],
                #Use seed 42 to ensure deterministic output
                options={'temperature': 0, 'seed': 42} 
            )
            match = re.search(r"0?\.\d+|1\.0|0|1", response['message']['content'])
            return float(match.group()) if match else 0.6
        except: 
            return 0.5

    def _generate_case_critique(self, student_plan, gt_plan, score, violations):
        if score >= 0.9: return "Excellent reasoning; matches expert human driver logic."
        prompt = (f"Explain in 1 sentence why this plan got {score}/1.0.\nGT: {gt_plan}\nStudent: {student_plan}\nViolations: {violations}")
        try:
            response = ollama.chat(
                model=self.judge_model, 
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0, 'seed': 42}
            )
            return response['message']['content'].strip().replace('"', '')
        except: return "Critique unavailable."

    def evaluate(self, white_response, ground_truth):
        report = {"id": ground_truth.get("id"), "scores": {}, "feedback": [], "generated_responses": white_response}
        
        perc = self._llm_score(white_response['perception'], ground_truth['perception'], "Perception")
        pred = self._llm_score(white_response['prediction'], ground_truth['prediction'], "Prediction")
        plan_raw = self._llm_score(white_response['planning'], ground_truth['planning'], "Planning")
        
        gt_context_str = f"{ground_truth['perception']} {ground_truth['planning']}"
        penalty, violations = check_safety_violation(white_response['planning'], gt_context_str)
        
        if perc < 0.2:
            plan_raw *= 0.6 
            report['feedback'].append("⚠️ Logic Warning: Plan penalized due to perception failure.")

        final_plan = max(0.0, plan_raw - (penalty * 0.5))
        critique = self._generate_case_critique(white_response['planning'], ground_truth['planning'], final_plan, violations)
        
        report['scores'] = {"perception": round(perc, 3), "prediction": round(pred, 3), "planning": round(final_plan, 3)}
        report['feedback'].extend(violations)
        report['violation_count'] = len(violations)
        report['critique'] = critique
        return report

    def compile_final_report(self, all_results, model_name="Agent"):
        if not all_results: return {"error": "No results"}

        avg_perc = statistics.mean([r['scores']['perception'] for r in all_results])
        avg_plan = statistics.mean([r['scores']['planning'] for r in all_results])
        total_violations = sum([r['violation_count'] for r in all_results])
        
        overall_percentage = round(((0.4 * avg_perc) + (0.3 * avg_plan) + (0.3 * statistics.mean([r['scores']['prediction'] for r in all_results]))) * 100, 2)

        if overall_percentage >= 90: grade = "A"
        elif overall_percentage >= 80: grade = "B"
        elif overall_percentage >= 70: grade = "C"
        elif overall_percentage >= 60: grade = "D"
        else: grade = "F"

        prompt = (
            f"You are evaluating a Vision-Language Model (VLM) for autonomous driving context named '{model_name}'.\n"
            f"STATS:\n"
            f"- Perception Score: {avg_perc:.2f}/1.0\n"
            f"- Planning Score: {avg_plan:.2f}/1.0\n"
            f"- Safety Violations: {total_violations}\n\n"
            "TASK: Write a JSON analysis with 3 keys: 'strengths', 'weaknesses', 'recommendations'.\n"
            "IMPORTANT Constraints:\n"
            "1. Recommendations must be about AI/ML (e.g., 'Increase Few-Shot examples', 'Tune Prompt Temperature').\n"
            "2. DO NOT suggest hardware (Lidar, Radar).\n"
            "3. If scores are low, suggest 'Chain-of-Thought prompting'.\n"
            "Example format: {\"strengths\": [\"...\"], \"weaknesses\": [\"...\"], \"recommendations\": [\"...\"]}"
        )

        try:
            response = ollama.chat(
                model=self.judge_model,
                messages=[{'role': 'user', 'content': prompt}],
                format="json",
                options={'temperature': 0, 'seed': 42}
            )
            analysis_json = json.loads(response['message']['content'])
            strengths = analysis_json.get('strengths', ["N/A"])
            weaknesses = analysis_json.get('weaknesses', ["N/A"])
            recommendations = analysis_json.get('recommendations', ["Adjust prompt engineering."])
        except Exception as e:
            strengths = ["Analysis failed."]
            weaknesses = [str(e)]
            recommendations = ["Retry."]

        return {
            "overall_grade": grade,
            "overall_score_percent": overall_percentage,
            "metrics": {
                "perception": round(avg_perc, 3),
                "prediction": round(statistics.mean([r['scores']['prediction'] for r in all_results]), 3),
                "planning": round(avg_plan, 3),
                "total_violations": total_violations
            },
            "analysis": {"strengths": strengths, "weaknesses": weaknesses, "recommendations": recommendations}
        }