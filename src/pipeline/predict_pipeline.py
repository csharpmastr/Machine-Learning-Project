import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.model_trainer import meta_learner_predictions

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            base_model_path = "artifacts\\base_model.pkl"
            meta_learner_path = 'artifacts\meta_learner.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            # load objects from file
            base_model=load_object(file_path=base_model_path)
            meta_learner=load_object(file_path=meta_learner_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            logging.info("Base Models, and Meta Learner has been loaded for prediction")
            
            # preprocess input data using preprocessor object
            scaled_data=preprocessor.transform(features)
            
            logging.info("User input has been preprocessed and ready for prediction")
            # make predictions on scaled data using the model object
            prediction = meta_learner_predictions(scaled_data, base_model, meta_learner)
            
            logging.info("Prediction has been made on user's input")
            
            prediction_list = prediction.tolist()
            logging.info("Prediction has been converted into a list object")
            
            print("Prediction before flatten:", prediction_list)
            
        #  prediction_base = np.argmax(prediction_list, axis=1)
            
        #     print("Prediction after flatten:", prediction_base)   
            
            interpretted_prediction = ""
            
            if prediction_list[0][0] == 1:
                interpretted_prediction = "Patient has No Diabetes"
            elif prediction_list[0][1] == 1:
                interpretted_prediction = "Patient is Pre-Diabetic"
            elif prediction_list[0][2] == 1:
                interpretted_prediction = "Patient is Diabetic"
            else:
                pass
            
            return interpretted_prediction
        except Exception as e:
            raise CustomException(e, sys)
        

# class CustomData:
#     def __init__(self,
#                  age: float,
#                  urea: float,
#                  cr: float,
#                  hbA1c: float,
#                  chol: float,
#                  tg: float,
#                  HDL: float,
#                  LDL: float,
#                  VLDL: float,
#                  BMI: float,
#                  Gender: str):
#         self.age = age
#         self.urea = urea
#         self.cr = cr
#         self.hbA1c = hbA1c
#         self.chol = chol
#         self.tg = tg
#         self.HDL = HDL,
#         self.LDL = LDL
#         self.VLDL = VLDL
#         self.BMI = BMI
#         self.Gender = Gender
    
#     def convert_data_as_dataframe(self):
#         try:
#             if any(value is None or (isinstance(value, str) and not value.strip()) for value in vars(self).values()):
#                 error_message = '<span style="color: red;font-size:25px;">Please fill in all the input fields.</span>'
#                 return error_message
            
#             custom_data_input = {
#                 "AGE": [self.age],
#                 "Urea": [self.urea],
#                 "Cr": [self.cr],
#                 "HbA1c": [self.hbA1c],
#                 "Chol": [self.chol],
#                 "TG": [self.tg],
#                 "HDL": [self.HDL],
#                 "LDL": [self.LDL],
#                 "VLDL": [self.VLDL],
#                 "BMI": [self.BMI],
#                 "Gender": [self.Gender]
#             }
            
#             return pd.DataFrame(custom_data_input)
#         except Exception as e:
#             raise CustomException(e, sys)
        
def diabetes_prediction(age: float,
                 urea: float,
                 cr: float,
                 hbA1c: float,
                 chol: float,
                 tg: float,
                 HDL: float,
                 LDL: float,
                 VLDL: float,
                 BMI: float,
                 Gender: str):
    
    try:
        # check if any of the input is empty
        if any([age is None, urea is None, cr is None, hbA1c is None, chol is None, 
                tg is None, HDL is None, LDL is None, VLDL is None, BMI is None, Gender is None]):
            error_message = '<span style="color: red;font-size:25px;">Please fill in all the input fields.</span>'
            return error_message

        # create dictionary of input data
        # custom_data_input = {
        #             "AGE": age,
        #             "Urea": urea,
        #             "Cr": cr,
        #             "HbA1c": hbA1c,
        #             "Chol": chol,
        #             "TG": tg,
        #             "HDL": HDL,
        #             "LDL": LDL,
        #             "VLDL": VLDL,
        #             "BMI": BMI,
        #             "Gender": Gender
        #         }
        gender_M = 0
        gender_F = 0
        if Gender == 'Male':
            gender_M = 1
        else:
            gender_F = 1
        
        custom_data_input = {
                    "AGE": [age],
                    "Urea": [urea],
                    "Cr": [cr],
                    "HbA1c": [hbA1c],
                    "Chol": [chol],
                    "TG": [tg],
                    "HDL": [HDL],
                    "LDL": [LDL],
                    "VLDL": [VLDL],
                    "BMI": [BMI],
                    "Gender_F": [gender_F],
                    "Gender_M": [gender_M]
                }
                
        input_df =  pd.DataFrame(custom_data_input)
        print(input_df.iloc[0:1, :])
        
        # make predictions
        predict_pipeline = PredictPipeline()
        prediction_result = predict_pipeline.predict(input_df)
        
        return prediction_result
    except Exception as e:
        raise CustomException(e, sys)