import os
from flask import Flask, render_template, request
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load the secret API key from the .env file
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the AI Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'data_file' not in request.files:
        return "No file uploaded.", 400
    
    file = request.files['data_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        import json
        user_prompt = request.form.get('user_prompt', '').strip()
        if not user_prompt:
            user_prompt = "Identify the two most critical business insights."
            
        df = pd.read_csv(filepath)
        metadata = df.dtypes.astype(str).to_dict()
        
        # --- THE STABLE DUAL-CHART PROMPT ---
        prompt = f"""
        You are an expert Enterprise Data Architect. 
        Metadata: {metadata}
        User Directive: "{user_prompt}"
        
        Suggest exactly TWO highly insightful charts that together form a comprehensive dashboard. 
        You MUST ONLY use the exact column names listed in the metadata. Do not guess or modify column names. Give 2 charts that together provide a holistic view of the data, ensuring they complement each other to reveal deeper insights. 
        Each chart should be distinct in type and perspective, avoiding redundancy. Avoid using ID columns as axes. Focus on actionable insights that can drive business decisions.
        Respond ONLY with a valid JSON object:
        {{
            "charts": [
                {{
                    "chart_type": "line/pie/scatter/bar/etc.", 
                    "x_axis": "column_name",
                    "y_axis": "column_name",
                    "aggregation": "sum/mean/etc.",
                    "rationale": "Explanation"
                }},
                {{
                    "chart_type": "line/pie/scatter/bar/etc.", 
                    "x_axis": "column_name",
                    "y_axis": "column_name",
                    "aggregation": "sum/mean/etc.",
                    "rationale": "Explanation"
                }}
            ]
        }}
        """
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
        )
        
        ai_response = json.loads(chat_completion.choices[0].message.content)
        charts_instructions = ai_response.get("charts", [])
        
        dashboard_data = []
        for i, chart in enumerate(charts_instructions):
            x_col, y_col = chart.get('x_axis'), chart.get('y_axis')
            if x_col in df.columns and y_col in df.columns:
                agg = chart.get('aggregation', 'mean')
                grouped = df.groupby(x_col)[y_col].mean().reset_index() if agg == 'mean' else df.groupby(x_col)[y_col].sum().reset_index()
                
                dashboard_data.append({
                    "id": f"chart-{i}",
                    "x": grouped[x_col].tolist(),
                    "y": grouped[y_col].tolist(),
                    "type": chart.get('chart_type', 'bar'),
                    "title": f"{agg.capitalize()} of {y_col} by {x_col}",
                    "rationale": chart.get('rationale')
                })
            
        return render_template('dashboard.html', dashboard_data=dashboard_data)
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)