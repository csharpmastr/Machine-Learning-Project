import sys
import gradio as gr
from src.exception import CustomException
from src.pipeline import predict_pipeline
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import diabetes_prediction

try:
    base_css = """.gradio-container {background-color: #FFC857}
                  #predict {background-color: #BDD9BF; 
                            border: solid 5px;
                            color: black;
                            -webkit-text-stroke: 1px white;
                            font-weight: bolder;
                            font-size: 25px}
                  #clear {background-color: red; 
                            border: solid 5px;
                            color: black;
                            -webkit-text-stroke: 1px white;
                            font-weight: bolder;
                            font-size: 25px}
                  #md {-webkit-text-stroke: 1px white}
                      """
    top_css = """<p style='text-align: center;
                font-size:60px;
                color: #2E4052;
                font-weight: Bolder'>Diabetes Prediction Test</p>"""
                
    bot_css = """<span style='text-align: center;
                justify-content: center;
                -webkit-text-stroke: 1px white;
                font-size:32px;
                color: #2E4052;
                font-weight: 800'>Patient Status</span>"""
    body_font = gr.themes.GoogleFont("Rubik")

    with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,font=body_font), css=base_css) as block:
        gr.Markdown(top_css, elem_id=["md"])
        
        with gr.Row():
            
            with gr.Column():
                with gr.Row():
                    gender = gr.Radio(["Male", "Female"], label="Gender", scale=1)
                with gr.Row():
                    age = gr.Number(label="Age", minimum=1, maximum=99)
                with gr.Row():
                    bmi = bmi = gr.Number(label="BMI", minimum=0)
                with gr.Row():
                    hba1c = gr.Number(label="HBA1C", minimum=0)
                with gr.Row():
                    cholesterol = gr.Number(label="Cholesterol", minimum=0)
                with gr.Row():
                    urea = gr.Number(label="Urea", minimum=0)
                
            
            with gr.Column():
                with gr.Row():
                    triglyceride = gr.Number(label="Triglyceride", minimum=0)
                with gr.Row():
                    hdl = gr.Number(label="High Dense Lipoprotein", minimum=0)
                with gr.Row():
                    ldl = gr.Number(label="Low Dense Lipoprotein", minimum=0)
                with gr.Row():
                    vldl = gr.Number(label="Very Low Dense Lipoprotein", minimum=0)
                with gr.Row():
                    creatinine = gr.Number(label="Creatinine", minimum=0)
            
                # creating variable to hold inputs
                input_data = [age, urea, creatinine, hba1c, cholesterol, triglyceride, hdl,
                            ldl, vldl, bmi, gender]
                
                with gr.Row():
                    with gr.Column():
                        gr.ClearButton(input_data, value="Clear", elem_id=["clear"], size='sm')                     
                    with gr.Column():
                        button = gr.Button("Predict and Analyze", elem_id=["predict"], size='sm')
                
        with gr.Row():
            with gr.Column():
                text = gr.Markdown(bot_css)
                output = gr.Textbox(placeholder="Awaiting Prediction", show_label=False, container=False)
                button.click(fn=diabetes_prediction, inputs=input_data, outputs=output) 
            
        
        gr.Examples([[57,3.0,60.0,7.9,4.8,2.4,1.8,2.0,1.1,27]],
                    inputs=input_data, label="Example Data")
            
        
        # with gr.Row():
        #     with gr.Row():
        #         gr.Textbox(label="Respond")
        #     with gr.Row():
        #         gr.Textbox(label="Prediction")
                          
                    
    block.launch(share=True)
except Exception as e:
    raise CustomException(e, sys)
             