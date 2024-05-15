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
                            font-weight: 800;
                            font-size: 30px}
                  #clear {background-color: red; 
                            border: solid 5px;
                            color: black;
                            -webkit-text-stroke: 1px white;
                            font-weight: 800;
                            font-size: 30px}
                  #md {-webkit-text-stroke: 1px white}
                  #output textarea {text-align: center; 
                            font-size: 32px;
                            background-color: #2E4052;
                            -webkit-text-stroke: 0.5px #BDD9BF;
                            font-weight: 800;
                            color: #000000}
                  #label {text-align: center;
                            justify-content: center;
                            -webkit-text-stroke: 1px white;
                            font-size:28px;
                            margin-top: 5%;
                            font-weight: 800}
                      """
    top_css = """<p style='text-align: center;
                font-size:60px;
                color: #2E4052;
                font-weight: 800'>Diabetes Prediction Test</p>"""
                
    # bot_css = """<span style='text-align: center;
    #             justify-content: center;
    #             -webkit-text-stroke: 1px white;
    #             font-size:32px;
    #             color: #2E4052;
    #             font-weight: 800'>Patient Status</span>"""
    # bot_css = """#label {text-align: center;
    #             justify-content: center;
    #             -webkit-text-stroke: 1px white;
    #             font-size:32px;
    #             color: #2E4052;
    #             margin-top: 10%;
    #             font-weight: 800}"""
                
    js_change_bg_color = """
                        function changeTextboxColor() {
                            var outputBox = document.getElementById("output");
                            
                            if (outputBox === "Patient has No Diabetes") {
                                outputBox.style.background-color = 'linear-gradient(to right, yellow, green)';
                            }
                        }
                         """
    body_font = gr.themes.GoogleFont("Rubik")
    
    def change_color(prediction):
        try:
            if prediction == "Patient has No Diabetes":
                return gr.Label(value="Patient has No Diabetes", color="#66CD3E", show_label=False, elem_id=["output"])
            elif prediction == "Patient is Pre-Diabetic":
                return gr.Label(value="Patient is Pre-Diabetic", color="#f15e2c", show_label=False, elem_id=["output"])
            elif prediction == "Patient is Diabetic":
                return gr.Label(value="Patient is Diabetic", color="#ff0030", show_label=False, elem_id=["output"])
            else:
                return gr.Label(value="No Prediction Made!", color="#ff0030", show_label=False, elem_id=["output"])
        except Exception as e:
            raise CustomException(e, sys)

    with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,font=body_font), css=base_css) as block:
        gr.Markdown(top_css, elem_id=["md"])
        
        with gr.Row():
            
            with gr.Column():
                with gr.Row():
                    gender = gr.Radio(["Male", "Female"], label="Gender", scale=1)
                with gr.Row():
                    age = gr.Number(label="Age", minimum=1, maximum=99, info="Age of the patient")
                with gr.Row():
                    bmi = bmi = gr.Number(label="BMI", minimum=0, maximum=49, info="BMI of the patient")
                with gr.Row():
                    hba1c = gr.Number(label="HBA1C", minimum=0, maximum=15, info="Hemoglobin A1c test in mmol/L")
                with gr.Row():
                    cholesterol = gr.Number(label="Cholesterol", minimum=0, maximum=20, info="Total Cholesterol of patient in mmol/L")
                with gr.Row():
                    urea = gr.Number(label="Urea", minimum=0, maximum=20, info="Patient's Blood Urea Nitrogen in mmol/L")
                
            
            with gr.Column():
                with gr.Row():
                    triglyceride = gr.Number(label="Triglyceride", minimum=0, maximum=20, info="Patient's Triglyceride Level in mmol/L (<1.69:Optimal), (1.69 to 2.25: Desirable), (2.26 to 5.64: High), (>5.65:Very High)")
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
                text = gr.Label(value="Patient Status", elem_id=["label"], show_label=False)
                # output = gr.Textbox(placeholder="Awaiting Prediction", show_label=False, elem_id=["output"], container=False, interactive=False)
                # output.change(fn=sample, js=js_change_bg_color, inputs=output)
                output = gr.Label(show_label=False, elem_id=["output"], color="66CD3E")
                predict = button.click(fn=diabetes_prediction, inputs=input_data, outputs=output) 
                output.change(fn=change_color, inputs=output, outputs=output)
            
        gr.Examples([[53,2.0,55.0,7.7,4.4,3.3,0.9,2.1,1.5,32],
                     [47,4.0,45,4.2,4.0,1.3,0.9,2.6,1.0,23.0],
                     [49,3.3,44,6.0,5.6,1.9,0.75,1.35,0.8,21.0]],
                    inputs=input_data, label="Example Data")
            
        
        # with gr.Row():
        #     with gr.Row():
        #         gr.Textbox(label="Respond")
        #     with gr.Row():
        #         gr.Textbox(label="Prediction")
                          
                    
    block.launch(share=True)
except Exception as e:
    raise CustomException(e, sys)
             