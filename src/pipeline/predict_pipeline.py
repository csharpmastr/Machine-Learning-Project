import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            # load objects from file
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            # preprocess input data using preprocessor object
            scaled_data=preprocessor.transform(features)
            
            # make predictions on scaled data using the model object
            prediction = model.predict(scaled_data)
            
            return prediction
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
                    "Gender": [Gender]
                }
                
        input_df =  pd.DataFrame(custom_data_input)
        gender_df = input_df['Gender']
        encoder = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False)
        
        # encode Gender column
        gender_df = encoder.fit_transform(gender_df)
        print(input_df)
        
        # make predictions
        predict_pipeline = PredictPipeline()
        prediction_result = predict_pipeline.predict(input_df)
        
        return prediction_result
    except Exception as e:
        raise CustomException(e, sys)