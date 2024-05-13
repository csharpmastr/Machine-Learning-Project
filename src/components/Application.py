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
                            font-size: 32px}
                  #clear {background-color: red; 
                            border: solid 5px;
                            color: black;
                            -webkit-text-stroke: 1px white;
                            font-weight: bolder;
                            font-size: 32px}
                  #md {-webkit-text-stroke: 1px white}
                      """
    top_css = """<p style='text-align: center;
                font-size:60px;
                color: #2E4052;
                font-weight: Bolder'>Diabetes Prediction Test</p>"""
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
                with gr.Row():
                    urea = gr.Number(label="Urea", minimum=0)
        
        # creating variable to hold inputs
        input_data = [age, urea, creatinine, hba1c, cholesterol, triglyceride, hdl,
                      ldl, vldl, bmi, gender]
        
        # instantiate Custom Data
        # data= CustomData(
        #     age=age,
        #     urea=urea,
        #     cr=creatinine,
        #     hbA1c=hba1c,
        #     chol=cholesterol, 
        #     tg=triglyceride,
        #     HDL=hdl,
        #     LDL=ldl, 
        #     VLDL=vldl,
        #     BMI=bmi,
        #     Gender=gender
        # )
        
        # convert data to dataframe
        # data_df = data.convert_data_as_dataframe()
        # print(data_df)
        
        # # make predictions from inputs
        # predict_pipe=PredictPipeline()
        # pred_result = predict_pipe.predict(data_df)
                
        with gr.Row():
            text = gr.Markdown("## Churn Status")
    
            # Define Gradio outputs
            output = gr.HTML("Awaiting Prediction")
            with gr.Column():
                button = gr.Button("Predict and Analyze", elem_id=["predict"])
                button.click(fn=diabetes_prediction, inputs=input_data, outputs=output) 
            with gr.Column():
                gr.ClearButton(input_data, value="Clear", elem_id=["clear"]) 
            
        
        # with gr.Row():
        #     with gr.Row():
        #         gr.Textbox(label="Respond")
        #     with gr.Row():
        #         gr.Textbox(label="Prediction")
                          
                    
    block.launch()
except Exception as e:
    raise CustomException(e, sys)
             