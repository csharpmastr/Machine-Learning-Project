import sys
import gradio as gr
import numpy
import requests
import json
import httpx
from src.exception import CustomException
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
            print(prediction)
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
                    age = gr.Number(label="Age", info="Age of the patient")
                with gr.Row():
                    height = gr.Number(label="Height", info="Height in centimeters")
                with gr.Row():
                    weight = gr.Number(label="Weight", info="Weight in Kilograms")   
                with gr.Row():
                    bmi = gr.Number(label="BMI", interactive=False) 
                with gr.Row():
                    hba1c = gr.Number(label="HBA1C", minimum=0, maximum=15, info="Hemoglobin A1c test in precentage(%) (5.7% - Normal), (5.7 to 6.4% - Pre-diabetes), (6.4% - Diabetes)")
                with gr.Row():
                    urea = gr.Number(label="Urea", minimum=0, maximum=20, info="Patient's Blood Urea Nitrogen in mmol/L")
                
            with gr.Column():
                with gr.Row():
                    triglyceride = gr.Number(label="Triglyceride", minimum=0, maximum=20, info="Patient's Triglyceride Level in mmol/L (<1.69:Optimal), (1.69 to 2.25: Desirable), (2.26 to 5.64: High), (>5.65:Very High)")
                with gr.Row():
                    hdl = gr.Number(label="High Density Lipoprotein", minimum=0, maximum=20, info="Patient's HDL level in mmol/L (<1.0 - Low), (1.0 to 1.5 - Desirable), (1.5 - High)")
                with gr.Row():
                    cholesterol = gr.Number(label="Cholesterol", minimum=0, maximum=20, info="Total Cholesterol of patient in mmol/L (<5.2 - Desirable), (5.2 to 6.1 - Borderline High), (6.1 - High)")
                with gr.Row():
                    ldl = gr.Number(label="Low Density Lipoprotein", minimum=0, maximum=20, info="Patient's LDL level in mmol/L (<2.6 - Optimal), (2.6 to 3.3 - Desirable), (3.4 to 4.0 - Borderline High), (4.1 to 4.8 - High), (4.8 - Very High)")
                with gr.Row():
                    vldl = gr.Number(label="Very Low Density Lipoprotein", minimum=0, maximum=20, info="Patient's VLDL in mmol/L")
                with gr.Row():
                    creatinine = gr.Number(label="Creatinine", minimum=0, info="Patient's Creatinine level in mmol/L (Creatinine: male adult: 60 to 110 - Normal) (female adult: 45 to 90 - Normal)")
                
                # function to establish api endpoints
                @gr.on(inputs=[age, weight, height], outputs=bmi)
                async def get_bmi(age, weight, height):
                    try:
                        data = [age, weight, height]
                        
                        no_zero = not any(var is None or var == 0 for var in data)
                        
                        if no_zero:
                            url = "http://127.0.0.1:8000/calculate_bmi"
                            
                            payload = {"age": data[0] if data else None,
                                        "weight": data[1] if len(data) > 1 else None, 
                                        "height": data[2] if len(data) > 2 else None}
                            
                            async with httpx.AsyncClient() as client:
                                response = await client.post(url, json=payload)
                                
                                while response.status_code == 200:
                                    try:
                                        json_data = response.json()
                                        bmi = json_data["bmi"]
                                        health = json_data["health"]
                                        print(f"BMI: {bmi}, Health: {health}")
                                        return bmi
                                    except json.JSONDecodeError:
                                        print("Error decoding JSON response")
                        else:
                            print("There are still none values:")
                            for i in data:
                                print(i)
                        
                    except Exception as e:
                        raise CustomException(e, sys)
                
                # variable to clear all inputs
                clear_data = [urea, creatinine, hba1c, cholesterol, triglyceride, hdl,
                                ldl, vldl, bmi, age, gender, height, weight]
                
                # creating variable to hold inputs
                input_data = [urea, creatinine, hba1c, cholesterol, triglyceride, hdl,
                                ldl, vldl, bmi, age, gender]
                
                with gr.Row():
                    with gr.Column():
                        gr.ClearButton(clear_data, value="Clear", elem_id=["clear"], size='lg')                     
                    with gr.Column():
                        button = gr.Button("Predict and Analyze", elem_id=["predict"], size='sm')
                        
        with gr.Row():
            with gr.Column():
                text = gr.Label(value="Patient Status", elem_id=["label"], show_label=False)
                output = gr.Label(show_label=False, elem_id=["output"], color="66CD3E")
                predict = button.click(fn=diabetes_prediction, inputs=input_data, outputs=output) 
                output.change(fn=change_color, inputs=output, outputs=output)
            
        gr.Examples([[2.0,55.0,7.7,4.4,3.3,0.9,2.1,1.5],
                     [4.0,45,4.2,4.0,1.3,0.9,2.6,1.0],
                     [3.3,44,6.0,5.6,1.9,0.75,1.35,0.8]],
                    inputs=input_data, label="Example Data")
                          
                    
    block.launch(share=True)
except Exception as e:
    raise CustomException(e, sys)
             