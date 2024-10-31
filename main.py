from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize FastAPI app
app = FastAPI()

# Define request body model
class ExperienceInput(BaseModel):
    YearsExperience: float

# Define route for prediction
@app.post("/predict")
async def predict_salary(input_data: ExperienceInput):
    experience_value = input_data.YearsExperience
    prediction = model.predict([[experience_value]])
    return {"predicted_salary": prediction[0]}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5000)