import os
import json
import base64

def encode_image_to_base64(image_path):
    """Reads an image and converts it to a base64 string."""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        if os.path.exists(os.path.abspath(image_path)):
             with open(os.path.abspath(image_path), "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        return ""
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def generate_leaderboard_report(json_path, output_html_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    sorted_agents = sorted(
        data.items(), 
        key=lambda x: x[1]['analysis']['overall_score_percent'], 
        reverse=True
    )
    
    # 1. Leaderboard Rows (with colored bars)
    leaderboard_rows = ""
    for rank, (agent_name, content) in enumerate(sorted_agents, 1):
        metrics = content['analysis']['metrics']
        score = content['analysis']['overall_score_percent']
        grade = content['analysis']['overall_grade']
        grade_color = "#27ae60" if grade == "PASS" else "#c0392b"
        
        # Helper for score bars
        def score_bar(val):
            color = "#f1c40f" # yellow
            if val > 0.7: color = "#27ae60" # green
            elif val < 0.4: color = "#c0392b" # red
            return f"""<div class="bar-container">
                         <div class="bar" style="width: {val*100}%; background: {color};"></div>
                       </div> {val}"""

        leaderboard_rows += f"""
            <tr onclick="openAgentTab('{agent_name}')" style="cursor: pointer;">
                <td>#{rank}</td>
                <td><b>{agent_name}</b></td>
                <td style="color: {grade_color}; font-weight:bold;">{grade}</td>
                <td><b>{score}%</b></td>
                <td>{score_bar(metrics['perception'])}</td>
                <td>{score_bar(metrics['prediction'])}</td>
                <td>{score_bar(metrics['planning'])}</td>
                <td style="color: {'#c0392b' if metrics['total_violations'] > 0 else 'inherit'}">{metrics['total_violations']}</td>
                <td>{round(sum(d['latency'] for d in content['details'])/len(content['details']), 2)}s</td>
            </tr>
        """

    # 2. Agent Tabs (with original insights layout)
    agent_tabs_html = ""
    tab_buttons = ""
    
    for idx, (agent_name, content) in enumerate(sorted_agents):
        active_class = "active" if idx == 0 else ""
        display_style = "block" if idx == 0 else "none"
        
        tab_buttons += f"""
            <button class="tab-link {active_class}" onclick="openAgentTab('{agent_name}')">{agent_name}</button>
        """

        analysis = content['analysis'].get('analysis', {})
        def safe_list(lst):
             return "".join([f"<li>{item}</li>" for item in (lst if lst else [])])

        score_color = "#27ae60" if content['analysis']['overall_grade'] == "PASS" else "#c0392b"

        # Test Cases (Collapsible)
        cases_html = ""
        for case in content['details']:
            case_id = case.get('id', 'unknown')
            img_path = case.get('image_path', '')
            img_b64 = encode_image_to_base64(img_path)
            
            img_elem = f'<img src="data:image/jpeg;base64,{img_b64}" class="case-img">' if img_b64 else '<div class="no-img">Image Not Found<br><small>' + img_path + '</small></div>'
            status_icon = "✅" if case['scores']['planning'] > 0.6 else "⚠️"
            critique = case.get('critique', 'No critique').replace('"', '')

            cases_html += f"""
            <details class="case-card">
                <summary class="case-header">
                    <span>{status_icon} <b>Test Case #{case_id}</b></span>
                    <span class="score-tag">Plan Score: {case['scores']['planning']}</span>
                </summary>
                <div class="case-body">
                    <div class="img-col">{img_elem}</div>
                    <div class="info-col">
                        <div class="log-row"><strong>👁️ Perception:</strong> {case['generated_responses'].get('perception', '-')}</div>
                        <div class="log-row"><strong>🔮 Prediction:</strong> {case['generated_responses'].get('prediction', '-')}</div>
                        <div class="log-row"><strong>🤖 Plan:</strong> {case['generated_responses'].get('planning', '-')}</div>
                        <div class="log-row ground-truth"><strong>📖 Truth Reference:</strong> {case['generated_responses'].get('gt_planning_context', '-')}</div>
                        <div class="log-row critique"><strong>📝 Judge:</strong> "{critique}"</div>
                    </div>
                </div>
            </details>
            """

        agent_tabs_html += f"""
        <div id="{agent_name}" class="agent-tab-content" style="display: {display_style};">
            
            <div class="agent-summary-header">
                <h2>Analysis: {agent_name}</h2>
                <div class="score-badge" style="color: {score_color}">
                    <div style="font-size: 1.5em;">{content['analysis']['overall_grade']}</div>
                    <span>Score: {content['analysis']['overall_score_percent']}%</span>
                </div>
            </div>

            <div class="insights-container">
                <div class="insight-box strength-box">
                    <h3>✅ Strengths</h3>
                    <ul>{safe_list(analysis.get('strengths'))}</ul>
                </div>
                <div class="insight-box weakness-box">
                    <h3>❌ Weaknesses</h3>
                    <ul>{safe_list(analysis.get('weaknesses'))}</ul>
                </div>
            </div>
            
            <div class="insight-box recommendation-box">
                <h3>💡 Recommendations</h3>
                <ul>{safe_list(analysis.get('recommendations'))}</ul>
            </div>

            <h3 style="margin-top:30px; border-bottom: 2px solid #eee; padding-bottom:10px;">Detailed Test Cases ({len(content['details'])})</h3>
            {cases_html}
        </div>
        """

    # 3. Final HTML (with restored CSS)
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚦 AutoDrive Benchmark</title>
        <style>
            body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 20px; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            
            header {{ text-align: center; margin-bottom: 30px; }}
            h1 {{ margin: 0; color: #2c3e50; }}
            
            /* Leaderboard */
            .leaderboard {{ overflow-x: auto; margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ background: #f8f9fa; color: #555; padding: 12px 15px; text-align: left; font-weight: 600; }}
            td {{ padding: 12px 15px; border-bottom: 1px solid #eee; vertical-align: middle; }}
            tr:hover {{ background: #f9f9f9; }}
            tr:last-child td {{ border-bottom: none; }}
            
            .bar-container {{ background: #ecf0f1; border-radius: 4px; height: 6px; width: 60px; display: inline-block; margin-right: 8px; vertical-align: middle; }}
            .bar {{ height: 100%; border-radius: 4px; }}

            /* Tabs */
            .tab-nav {{ display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 0; }}
            .tab-link {{ padding: 10px 20px; border: none; background: transparent; border-bottom: 3px solid transparent; cursor: pointer; font-weight: 600; color: #777; transition: all 0.2s; }}
            .tab-link:hover {{ color: #333; }}
            .tab-link.active {{ border-bottom: 3px solid #3498db; color: #3498db; }}
            
            /* Agent Summary Header */
            .agent-summary-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; }}
            .agent-summary-header h2 {{ margin: 0; color: #2c3e50; }}
            .score-badge {{ text-align: right; font-weight: bold; }}
            
            /* Insights Layout (Restored) */
            .insights-container {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .insight-box {{ flex: 1; padding: 20px; border-radius: 8px; }}
            .insight-box h3 {{ margin-top: 0; margin-bottom: 15px; font-size: 1.1em; display: flex; align-items: center; }}
            .insight-box ul {{ padding-left: 20px; margin: 0; line-height: 1.6; }}
            
            .strength-box {{ background: #e8f5e9; color: #1b5e20; }}
            .weakness-box {{ background: #ffebee; color: #b71c1c; }}
            .recommendation-box {{ background: #e3f2fd; color: #0d47a1; }}

            /* Collapsible Cases */
            details.case-card {{ background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 15px; overflow: hidden; border: 1px solid #e0e0e0; }}
            summary.case-header {{ background: #f8f9fa; padding: 12px 20px; display: flex; justify-content: space-between; font-weight: bold; cursor: pointer; list-style: none; user-select: none; }}
            summary.case-header::-webkit-details-marker {{ display: none; }}
            summary.case-header:hover {{ background: #e9ecef; }}
            
            .case-body {{ display: flex; padding: 20px; gap: 20px; border-top: 1px solid #eee; animation: fadeIn 0.2s ease-in-out; }}
            @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

            .img-col {{ flex: 0 0 300px; }}
            .case-img {{ width: 100%; border-radius: 6px; border: 1px solid #eee; }}
            .no-img {{ width: 100%; height: 180px; background: #eee; display: flex; align-items: center; justify-content: center; text-align: center; color: #777; border-radius: 6px; }}
            
            .info-col {{ flex: 1; }}
            .log-row {{ margin-bottom: 10px; font-size: 0.95em; line-height: 1.5; }}
            .ground-truth {{ background: #fffde7; padding: 12px; border-left: 4px solid #f1c40f; border-radius: 4px; margin-top: 15px; }}
            .critique {{ background: #e8f6fa; padding: 12px; border-left: 4px solid #3498db; border-radius: 4px; font-style: italic; margin-top: 15px; }}
            
        </style>
        <script>
            function openAgentTab(agentName) {{
                var i;
                var x = document.getElementsByClassName("agent-tab-content");
                for (i = 0; i < x.length; i++) {{ x[i].style.display = "none"; }}
                
                var tablinks = document.getElementsByClassName("tab-link");
                for (i = 0; i < x.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
                
                document.getElementById(agentName).style.display = "block";
                
                var btns = document.getElementsByTagName("button");
                for (i = 0; i < btns.length; i++) {{
                    if(btns[i].innerText === agentName) {{
                        btns[i].className += " active";
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🚦 AutoDrive Benchmark Results</h1>
                <p style="color: #777;">Generated: {os.popen('date').read().strip()}</p>
            </header>
            
            <h2 style="margin-top:0;">🏆 Leaderboard</h2>
            <div class="leaderboard">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th><th>Model</th><th>Grade</th><th>Score</th>
                            <th>Perception</th><th>Prediction</th><th>Planning</th><th>Violations</th><th>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {leaderboard_rows}
                    </tbody>
                </table>
            </div>

            <div class="tab-nav">
                {tab_buttons}
            </div>

            {agent_tabs_html}
        </div>
    </body>
    </html>
    """
    
    with open(output_html_path, 'w') as f:
        f.write(html_template)
    
    return output_html_path