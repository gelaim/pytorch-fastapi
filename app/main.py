from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.trained import predict

app = FastAPI()

class TextIn(BaseModel):
    txt: str

class TextOut(TextIn):
    txt: str


# routes

@app.get("/")
async def hello():
    return {"oi": "boi!"}


@app.post("/predict", response_model=str, status_code=200)
def get_prediction(payload: TextIn):
    txt = payload.txt
    
    prediction_list = predict(txt)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")
    #adict= {}
    #adict['response'] = ''
    #for i in prediction_list:
    #    adict['response']+= i

    return prediction_list
