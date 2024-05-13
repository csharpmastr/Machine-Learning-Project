import sys
import gradio as gr
from src.exception import CustomException

try:
    base_css = """.gradio-container {background: #FFC857;}"""
    top_css = """<p style='text-align: center;
                font-size:60px;
                color: #2E4052;
                font-weight: Bolder'>Diabetes Prediction Test</p>"""
    body_font = gr.themes.GoogleFont("Rubik")

    with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,font=body_font), css=base_css) as block:
        gr.Markdown(top_css)
        
        with gr.Row():
            
            with gr.Column():
                with gr.Row():
                    gender = gr.Radio(["Male", "Female"], label="Gender")
                with gr.Row():
                    age = gr.Number(label="Age", minimum=1, maximum=99)
                with gr.Row():
                    bmi = bmi = gr.Number(label="BMI", minimum=0)
                with gr.Row():
                    hba1c = gr.Number(label="HBA1C", minimum=0)
                with gr.Row():
                    cholesterol = gr.Number(label="Cholesterol", minimum=0)
                
            
            with gr.Column():
                with gr.Row():
                    triglyceride = gr.Number(label="Triglyceride", minimum=0)
                with gr.Row():
                    high_dense_lipoprotein = gr.Number(label="High Dense Lipoprotein", minimum=0)
                with gr.Row():
                    low_dense_lipoprotein = gr.Number(label="Low Dense Lipoprotein", minimum=0)
                with gr.Row():
                    creatinine = gr.Number(label="Creatinine", minimum=0)
                with gr.Row():
                    urea = gr.Number(label="Urea", minimum=0)
                
        with gr.Row():
            with gr.Column():
                button = gr.Button("Predict and Analyze")
                button.click() 
            with gr.Column():
                clear = gr.ClearButton(value="Clear") 
            
        
        with gr.Row():
            with gr.Row():
                gr.Textbox(label="Respond")
            with gr.Row():
                gr.Textbox(label="Prediction")
                          
                    
    block.launch()
except Exception as e:
    raise CustomException(e, sys)
             