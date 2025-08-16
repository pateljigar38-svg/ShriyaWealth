from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/predict")
def predict(query: Query):
    # Example logic, replace with your fund advisor logic
    return {"answer": f"You asked: {query.question}"}
