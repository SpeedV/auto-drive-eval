import json
import datetime
import os

def generate_leaderboard_report(json_path, html_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Ranking Calculation
    ranking_data = []
    for model_name, content in data.items():
        metrics = content['analysis']['metrics']
        overall = content['analysis']['overall_score_percent']
        latencies = [d.get('latency', 0) for d in content['details']]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

        ranking_data.append({
            "name": model_name,
            "score": overall,
            "grade": content['analysis']['overall_grade'],
            "perc": metrics['perception'],
            "pred": metrics['prediction'],
            "plan": metrics['planning'],
            "safety": metrics['total_violations'],
            "latency": avg_latency
        })

    ranking_data.sort(key=lambda x: x['score'], reverse=True)

    # 2. HTML Structure
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AutoDrive Benchmark</title>
        <style>
            body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 20px; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .leaderboard-card {{ background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ text-align: left; padding: 12px; border-bottom: 2px solid #eee; color: #555; }}
            td {{ padding: 12px; border-bottom: 1px solid #eee; }}
            .rank-1 {{ background-color: #fff9c4; font-weight: bold; }}
            
            .tabs {{ display: flex; gap: 10px; margin-bottom: 0; }}
            .tab-btn {{ padding: 12px 24px; background: #e0e0e0; border: none; border-radius: 8px 8px 0 0; cursor: pointer; font-weight: 600; transition: 0.2s; }}
            .tab-btn.active {{ background: white; color: #2196f3; box-shadow: 0 -2px 5px rgba(0,0,0,0.05); }}
            
            .model-content {{ display: none; background: white; padding: 25px; border-radius: 0 8px 8px 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .model-content.active {{ display: block; }}
            
            .analysis-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
            .rec-panel {{ grid-column: span 2; background: #e3f2fd; color: #1565c0; }}
            .panel {{ padding: 15px; border-radius: 6px; }}
            .panel h4 {{ margin-top: 0; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 8px; }}
            .good {{ background: #e8f5e9; color: #2e7d32; }}
            .bad {{ background: #ffebee; color: #c62828; }}
            
            details {{ margin-bottom: 15px; border: 1px solid #ddd; border-radius: 6px; overflow: hidden; background: #fff; }}
            summary {{ padding: 15px; cursor: pointer; background: #f9f9f9; font-weight: 600; display: flex; justify-content: space-between; align-items: center; }}
            .details-body {{ padding: 20px; border-top: 1px solid #ddd; }}
            
            .case-grid {{ display: flex; gap: 20px; }}
            .img-box {{ width: 250px; background: #eee; }}
            .img-box img {{ width: 100%; height: auto; }}
            .text-box {{ flex: 1; }}
            
            .agent-output {{ background: #fafafa; padding: 10px; border: 1px solid #eee; border-radius: 4px; margin-bottom: 10px; }}
            .agent-label {{ font-size: 0.8em; font-weight: bold; color: #777; text-transform: uppercase; margin-bottom: 4px; display: block; }}
            
            .critique {{ margin-top: 15px; padding: 10px; background: #fff3e0; border-left: 4px solid #ff9800; font-style: italic; color: #666; }}
            .bar-bg {{ width: 60px; height: 6px; background: #ddd; border-radius: 3px; display: inline-block; }}
            .bar-fg {{ height: 100%; border-radius: 3px; }}
        </style>
        <script>
            function openModel(modelName) {{
                var contents = document.getElementsByClassName("model-content");
                for (var i = 0; i < contents.length; i++) {{ contents[i].classList.remove("active"); }}
                var btns = document.getElementsByClassName("tab-btn");
                for (var i = 0; i < btns.length; i++) {{ btns[i].classList.remove("active"); }}
                document.getElementById(modelName).classList.add("active");
                document.getElementById("btn-" + modelName).classList.add("active");
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🚦 AutoDrive Benchmark Results</h1>
            <p>Generated: {date_str}</p>
            
            <div class="leaderboard-card">
                <h2>🏆 Leaderboard</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Grade</th>
                            <th>Score</th>
                            <th>Perception</th>
                            <th>Prediction</th> <th>Planning</th>
                            <th>Violations</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for idx, r in enumerate(ranking_data):
        rank_cls = "rank-1" if idx == 0 else ""
        p_color = "#4caf50" if r['perc'] > 0.6 else "#ff9800"
        pr_color = "#4caf50" if r['pred'] > 0.6 else "#ff9800"
        pl_color = "#4caf50" if r['plan'] > 0.6 else "#ff9800"
        
        html += f"""
            <tr class="{rank_cls}">
                <td>#{idx+1}</td>
                <td style="font-weight:bold">{r['name']}</td>
                <td>{r['grade']}</td>
                <td style="font-size:1.1em; font-weight:bold">{r['score']}%</td>
                <td><div class="bar-bg"><div class="bar-fg" style="width:{r['perc']*100}%; background:{p_color}"></div></div> {r['perc']}</td>
                <td><div class="bar-bg"><div class="bar-fg" style="width:{r['pred']*100}%; background:{pr_color}"></div></div> {r['pred']}</td>
                <td><div class="bar-bg"><div class="bar-fg" style="width:{r['plan']*100}%; background:{pl_color}"></div></div> {r['plan']}</td>
                <td style="color:{'red' if r['safety']>0 else 'green'}">{r['safety']}</td>
                <td>{r['latency']}s</td>
            </tr>
        """
        
    html += """</tbody></table></div><div class="tabs">"""
    for idx, r in enumerate(ranking_data):
        active = "active" if idx == 0 else ""
        html += f"""<button id="btn-{r['name']}" class="tab-btn {active}" onclick="openModel('{r['name']}')">{r['name']}</button>"""
    html += "</div>"

    for idx, r in enumerate(ranking_data):
        model_name = r['name']
        data_node = data[model_name]
        analysis = data_node['analysis']
        active = "active" if idx == 0 else ""
        
        html += f"""
        <div id="{model_name}" class="model-content {active}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2>Analysis: {model_name}</h2>
                <div style="text-align:right"><span style="font-size:2em; font-weight:bold;">{analysis['overall_grade']}</span><br><span style="color:#777">Score: {analysis['overall_score_percent']}%</span></div>
            </div>
            <div class="analysis-grid">
                <div class="panel good"><h4>✅ Strengths</h4><ul style="padding-left:20px; margin:0">{''.join(f'<li>{x}</li>' for x in analysis['analysis']['strengths'])}</ul></div>
                <div class="panel bad"><h4>❌ Weaknesses</h4><ul style="padding-left:20px; margin:0">{''.join(f'<li>{x}</li>' for x in analysis['analysis']['weaknesses'])}</ul></div>
                <div class="panel rec-panel"><h4>💡 Recommendations</h4><ul style="padding-left:20px; margin:0">{''.join(f'<li>{x}</li>' for x in analysis['analysis']['recommendations'])}</ul></div>
            </div>
            <h3>Test Cases ({len(data_node['details'])})</h3>
        """
        
        for item in data_node['details']:
            img_id = item['id']
            img_src = f"../dataset/images/{img_id:03d}.jpg"
            plan_score = item['scores']['planning']
            icon = "✅" if plan_score > 0.6 else "⚠️"
            violations = f"<div style='margin-top:10px; color:#d32f2f; font-weight:bold;'>Violations: {', '.join(item['feedback'])}</div>" if item['feedback'] else ""

            html += f"""
            <details>
                <summary><span>{icon} Test Case #{img_id}</span><span>Plan Score: {plan_score}</span></summary>
                <div class="details-body">
                    <div class="case-grid">
                        <div class="img-box"><img src="{img_src}" loading="lazy" onerror="this.src='https://placehold.co/300x200?text=No+Image'"></div>
                        <div class="text-box">
                            <div class="agent-output">
                                <span class="agent-label">👁️ Perception</span>
                                {item['generated_responses']['perception']}
                            </div>
                            <div class="agent-output">
                                <span class="agent-label">🔮 Prediction</span>
                                {item['generated_responses']['prediction']}
                            </div>
                            <div class="agent-output" style="border-left: 3px solid #2196f3;">
                                <span class="agent-label">🤖 Plan</span>
                                {item['generated_responses']['planning']}
                            </div>
                            <div style="margin-top:10px; font-size:0.9em; color:#555">
                                <strong>📖 Ground Truth Reference:</strong> {item['generated_responses'].get('gt_planning_context', 'See JSON')}
                            </div>
                            {violations}
                            <div class="critique"><strong>📝 Judge's Critique:</strong><br>"{item.get('critique', 'N/A')}"</div>
                        </div>
                    </div>
                </div>
            </details>
            """
        html += "</div>"
    html += "</div></body></html>"
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)