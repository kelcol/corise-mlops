from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import datetime
import sys

from classifier import NewsCategoryClassifier

MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"

logger.add(LOGS_OUTPUT_PATH)


app = FastAPI()
nc = None


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str

class LogRequestResponse(BaseModel):
    timestamp: str = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    request: dict
    prediction: dict
    latency: str
    


@app.on_event("startup")
def startup_event():
    global nc
    nc = NewsCategoryClassifier(verbose=True)
    nc.load(MODEL_PATH)
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    logger.info("Shutting down application")
    with open(LOGS_OUTPUT_PATH, "w+") as f:
        f.flush()
        f.close()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    start = datetime.datetime.now()
    if nc == None:
        startup_event()

    # get model prediction for the input request
    predictions = nc.run_prediction(request)
    sorted_predictions = dict(sorted(predictions.items(), key=lambda item:item[1], reverse=True))    
    label = list(sorted_predictions)[0]
    
    end = datetime.datetime.now() 
    latency = (end-start).total_seconds() * 1000

    # construct the data to be logged
    logger.info(LogRequestResponse(request=request.dict(), prediction=sorted_predictions, latency=latency).json())

    # construct response
    return PredictResponse(scores=sorted_predictions, label=label)


@app.get("/")
def read_root():
    return {"Hello": "World"}