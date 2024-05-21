from fastapi import FastAPI
from pydantic import BaseModel
import sys
from src.exception import CustomException

# instantiate FastAPI
app = FastAPI()

class BMICalculationRequest(BaseModel):
    age: int
    height: float  
    weight: float  

class BMICalculationResponse(BaseModel):
    bmi: float
    health: str
    healthy_bmi_range: str

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi <= 24.9:
        return "Normal"
    elif 25.0 <= bmi <= 29.9:
        return "Overweight"
    else:
        return "Obese"
    
@app.post("/calculate_bmi", response_model=BMICalculationResponse)
def calculate_bmi(request: BMICalculationRequest):
    try:
        height_meters = request.height / 100
        bmi = round(request.weight / (height_meters ** 2), 2)
        health = get_bmi_category(bmi)
        healthy_bmi_range = "18.5 - 24.9"

        return BMICalculationResponse(bmi=bmi, health=health, healthy_bmi_range=healthy_bmi_range)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)