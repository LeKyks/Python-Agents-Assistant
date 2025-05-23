"""
Simple UI utilities for the agents system
"""
import gradio as gr
import requests
import os
import json

# Configuration de l'API
API_URL = "http://localhost:8000"

def generate_readme(project_name, project_description, technologies, code_snippets, sections):
    """Generate a README using the API"""
    tech_list = [t.strip() for t in technologies.split(',') if t.strip()]
    code_list = code_snippets.split('```')
    code_list = [c.strip() for c in code_list if c.strip()]
    section_list = [s.strip() for s in sections.split(',') if s.strip()]
    
    data = {
        "project_name": project_name,
        "project_description": project_description,
        "technologies": tech_list,
        "code_snippets": code_list,
        "include_sections": section_list
    }
    
    response = requests.post(f"{API_URL}/readme/generate", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

def improve_code(code, task_type, requirements, context):
    """Improve code using the API"""
    req_list = [r.strip() for r in requirements.split('\n') if r.strip()]
    
    data = {
        "code": code,
        "task_type": task_type,
        "requirements": req_list,
        "context": context
    }
    
    response = requests.post(f"{API_URL}/code/improve", json=data)
    if response.status_code == 200:
        result = response.json()
        return result["improved_code"], result["explanation"]
    else:
        return f"Error: {response.status_code}", response.text

def debug_code(code, error_message):
    """Debug code using the API"""
    data = {
        "code": code,
        "error_message": error_message
    }
    
    response = requests.post(f"{API_URL}/code/debug", json=data)
    if response.status_code == 200:
        result = response.json()
        return result["debug_report"], result["fixed_code"]
    else:
        return f"Error: {response.status_code}", response.text

def process_document(file):
    """Process a document for RAG"""
    if not file:
        return "No file provided"
    
    files = {"file": (os.path.basename(file), open(file, "rb"))}
    response = requests.post(f"{API_URL}/rag/process", files=files)
    
    if response.status_code == 200:
        result = response.json()
        return json.dumps(result, indent=2)
    else:
        return f"Error: {response.status_code} - {response.text}"

def query_document(query, index_path=None):
    """Query a processed document"""
    data = {
        "query": query,
        "index_path": index_path,
        "operation": "query"
    }
    
    response = requests.post(f"{API_URL}/rag/query", json=data)
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            return result["answer"]
        else:
            return f"Error: {result['message']}"
    else:
        return f"Error: {response.status_code} - {response.text}"

def create_ui():
    """Create Gradio UI for the agents"""
    with gr.Blocks(title="Python Agents Assistant") as app:
        gr.Markdown("# Python Agents Assistant")
        
        with gr.Tab("README Generator"):
            with gr.Row():
                with gr.Column():
                    project_name = gr.Textbox(label="Project Name")
                    project_desc = gr.Textbox(label="Project Description", lines=3)
                    technologies = gr.Textbox(label="Technologies (comma separated)", lines=2)
                    code_snippets = gr.Code(label="Code Snippets (separate with ```)", language="python", lines=10)
                    sections = gr.Textbox(
                        label="Sections to Include (comma separated)",
                        value="Introduction, Installation, Usage, Features, Technologies, Project Structure, Contributing, License",
                        lines=2
                    )
                    readme_btn = gr.Button("Generate README")
                
                with gr.Column():
                    readme_output = gr.Markdown(label="Generated README")
        
        with gr.Tab("Code Assistant"):
            with gr.Row():
                with gr.Column():
                    code_input = gr.Code(label="Your Code", language="python", lines=15)
                    task_type = gr.Dropdown(
                        choices=["correction", "optimisation", "refactoring", "pep8", "debug"],
                        value="correction",
                        label="Task Type"
                    )
                    requirements = gr.Textbox(label="Requirements (one per line)", lines=3)
                    context = gr.Textbox(label="Code Context", lines=2)
                    code_btn = gr.Button("Improve Code")
                
                with gr.Column():
                    improved_code = gr.Code(label="Improved Code", language="python", lines=15)
                    explanation = gr.Textbox(label="Explanation", lines=10)
        
        with gr.Tab("Debug Assistant"):
            with gr.Row():
                with gr.Column():
                    debug_code_input = gr.Code(label="Code to Debug", language="python", lines=15)
                    error_message = gr.Textbox(label="Error Message (optional)", lines=3)
                    debug_btn = gr.Button("Debug Code")
                
                with gr.Column():
                    debug_report = gr.Textbox(label="Debug Report", lines=10)
                    fixed_code = gr.Code(label="Fixed Code", language="python", lines=15)
        
        with gr.Tab("Document Q&A (RAG)"):
            with gr.Row():
                with gr.Column():
                    document_file = gr.File(label="Upload Document")
                    process_btn = gr.Button("Process Document")
                    process_result = gr.Textbox(label="Processing Result", lines=5)
                
                with gr.Column():
                    query_input = gr.Textbox(label="Your Question", lines=2)
                    index_path = gr.Textbox(label="Index Path (optional)", lines=1)
                    query_btn = gr.Button("Ask Question")
                    answer_output = gr.Textbox(label="Answer", lines=15)
        
        # Wire up the buttons
        readme_btn.click(
            generate_readme,
            inputs=[project_name, project_desc, technologies, code_snippets, sections],
            outputs=readme_output
        )
        
        code_btn.click(
            improve_code,
            inputs=[code_input, task_type, requirements, context],
            outputs=[improved_code, explanation]
        )
        
        debug_btn.click(
            debug_code,
            inputs=[debug_code_input, error_message],
            outputs=[debug_report, fixed_code]
        )
        
        process_btn.click(
            process_document,
            inputs=[document_file],
            outputs=[process_result]
        )
        
        query_btn.click(
            query_document,
            inputs=[query_input, index_path],
            outputs=[answer_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()
