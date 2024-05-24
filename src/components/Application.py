import sys
from tkinter import Label
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
                  #gender {height: 100px}
                  #bmi {text-align: center;
                        justify-content: center}
                  #bmi input {text-align: center;
                        justify-content: center}
                  #unit {text-align: center;
                        justify-content: center}
                  .radio-group .wrap {display: grid;
                            grid-template-columns: 1fr 1fr;
                            align-items: center}
                      """
    top_css = """<p style='text-align: center;
                font-size:60px;
                color: #2E4052;
                font-weight: 800'>Diabetes Prediction Test</p>"""
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
    
    def change_unit_measurement(unit):
        if unit == "mg/dL":
            return  [
                    gr.Number(label="Cholesterol", info="Total Cholesterol of patient in mg/dL (<5.2 - Desirable), (5.2 to 6.1 - Borderline High), (6.1 - High)"),
                    gr.Number(label="Low Density Lipoprotein", info="Patient's LDL level in mg/dL (<2.6 - Optimal), (2.6 to 3.3 - Desirable), (3.4 to 4.0 - Borderline High), (4.1 to 4.8 - High), (4.8 - Very High)"),
                    gr.Number(label="High Density Lipoprotein", info="Patient's HDL level in mg/dL (<1.0 - Low), (1.0 to 1.5 - Desirable), (1.5 - High)"),
                    gr.Number(label="Very Low Density Lipoprotein", info="Patient's VLDL in mg/dL"),
                    gr.Number(label="Triglyceride", info="Patient's Triglyceride Level in mg/dL (1.69 to 2.25: Desirable)"),
                    gr.Number(label="Creatinine", info="Patient's Creatinine level in mg/dL (Normal Range for Male Adult: 60 to 110) (Normal Range for Female Adult: 45 to 90 - Normal)"),
                    gr.Number(label="Urea", info="Patient's Blood Urea Nitrogen in mg/dL")
            ]
        else:
            return [
                    gr.Number(label="Cholesterol", info="Total Cholesterol of patient in mmol/L (<5.2 - Desirable), (5.2 to 6.1 - Borderline High), (6.1 - High)"),
                    gr.Number(label="Low Density Lipoprotein", info="Patient's LDL level in mmol/L (<2.6 - Optimal), (2.6 to 3.3 - Desirable), (3.4 to 4.0 - Borderline High), (4.1 to 4.8 - High), (4.8 - Very High)"),
                    gr.Number(label="High Density Lipoprotein", info="Patient's HDL level in mmol/L (<1.0 - Low), (1.0 to 1.5 - Desirable), (1.5 - High)"),
                    gr.Number(label="Very Low Density Lipoprotein", info="Patient's VLDL in mmol/L"),
                    gr.Number(label="Triglyceride", info="Patient's Triglyceride Level in mmol/L (1.69 to 2.25: Desirable)"),
                    gr.Number(label="Creatinine", info="Patient's Creatinine level in mmol/L (Normal Range for Male Adult: 60 to 110) (Normal Range for Female Adult: 45 to 90 - Normal)"),
                    gr.Number(label="Urea", info="Patient's Blood Urea Nitrogen in mmol/L")

            ]

    with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,font=body_font), css=base_css) as block:
        gr.Markdown(top_css, elem_id=["md"])
        
        with gr.Row():
            
            with gr.Column():
                with gr.Row():
                    gender = gr.Radio(["Male", "Female"], label="Gender", scale=2, elem_id=['gender'], info="Gender of the Patient")
                with gr.Row():
                    height = gr.Number(label="Height", info="Height in centimeters")
            with gr.Column():
                with gr.Row():
                    age = gr.Number(label="Age", info="Age of the patient", scale=1)
                with gr.Row():
                    weight = gr.Number(label="Weight", info="Weight in Kilograms")  
                     
        with gr.Row():
                    bmi = gr.Number(label="BMI", interactive=False, elem_id=['bmi'], info="Patient's Body Mass Index (<18.5: Underweight) (18.5 - 24.9: Normal) (25.0 - 29.9: Overweight) (>29.9: Obese)") 
        
        with gr.Row():
                    unit = gr.Radio(choices=["mmol/L", "mg/dL"], elem_classes=['radio-group'], elem_id=['unit'], value="mmol/L", label="Unit of measurement", info="Unit of measurement either milligrams per deciliter (mg/dL) or millimoles per litre (mmol/L)")
        
        with gr.Row():
            with gr.Column():                                    
                with gr.Row():
                    cholesterol = gr.Number(label="Cholesterol", info="Total Cholesterol of patient in mmol/L (<5.2 - Desirable), (5.2 to 6.1 - Borderline High), (6.1 - High)")
                with gr.Row():
                    ldl = gr.Number(label="Low Density Lipoprotein", info="Patient's LDL level in mmol/L (<2.6 - Optimal), (2.6 to 3.3 - Desirable), (3.4 to 4.0 - Borderline High), (4.1 to 4.8 - High), (4.8 - Very High)")
                with gr.Row():
                    hdl = gr.Number(label="High Density Lipoprotein", info="Patient's HDL level in mmol/L (<1.0 - Low), (1.0 to 1.5 - Desirable), (1.5 - High)")
                with gr.Row():
                    vldl = gr.Number(label="Very Low Density Lipoprotein", info="Patient's VLDL in mmol/L (0.31 to 0.78 - Desirable)")
                      
            with gr.Column():
                with gr.Row():
                    triglyceride = gr.Number(label="Triglyceride", info="Patient's Triglyceride Level in mmol/L (1.69 to 2.25: Desirable)")
                with gr.Row():
                    creatinine = gr.Number(label="Creatinine", info="Patient's Creatinine level in mmol/L (Normal Range for Male Adult: 60 to 110) (Normal Range for Female Adult: 45 to 90 - Normal)")
                with gr.Row():
                    hba1c = gr.Number(label="HBA1C", info="Hemoglobin A1c test in precentage(%) (5.7% - Normal), (5.7 to 6.4% - Pre-diabetes), (6.4% - Diabetes)")
                with gr.Row():
                    urea = gr.Number(label="Urea", info="Patient's Blood Urea Nitrogen in mmol/L")
                
                # on unit change
                unit.change(fn=change_unit_measurement, inputs=unit, outputs=[cholesterol, ldl, hdl, vldl, triglyceride, creatinine, urea])
                
                # function to establish api endpoints
                @gr.on(inputs=[age, weight, height], outputs=bmi)
                async def get_bmi(age, weight, height):
                    try:
                        data = [age, weight, height]
                        
                        # check if data has None or 0 values
                        no_zero = not any(var is None or var == 0 for var in data)
                        
                        # establish a connection with the api's server
                        if no_zero:
                            # define api endpoint
                            url = "http://127.0.0.1:8000/calculate_bmi"
                            
                            # prepare data to send in the POST request
                            payload = {"age": data[0] if data else None,
                                        "weight": data[1] if len(data) > 1 else None, 
                                        "height": data[2] if len(data) > 2 else None}
                            
                            # create async HTTP client
                            async with httpx.AsyncClient() as client:
                                response = await client.post(url, json=payload)
                                
                                # checking if the response is OK
                                while response.status_code == 200:
                                    try:
                                        # parse json response
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
                                ldl, vldl, bmi, age, gender, unit]
                
        with gr.Row():
            with gr.Column():
                gr.ClearButton(clear_data, value="Clear", elem_id=["clear"], size='lg')                     
            with gr.Column():
                button = gr.Button("Predict and Analyze", elem_id=["predict"], size='lg')
                        
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
             