from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import datetime

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
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    start = datetime.datetime.now()
    preds = nc.predict_proba(request)
    
    sorted_preds = dict(sorted(preds.items(), key=lambda item:item[1], reverse=True))
    label = list(sorted_preds)[0]    
    response = PredictResponse(scores=sorted_preds, label=label)

    end = datetime.datetime.now()
    msg = {}
    msg['timestamp'] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    msg['request'] = request.dict()
    msg['prediction'] = response.dict()    
    diff = end-start    
    latency = diff.total_seconds() * 1000
    msg['latency'] = latency
    logger.info(msg)

    return response


@app.post("/predict_label", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    start = datetime.datetime.now()
    preds = nc.predict_proba(request)
    
    sorted_preds = dict(sorted(preds.items(), key=lambda item:item[1], reverse=True))
    label = list(sorted_preds)[0]    
    response = PredictResponse(scores={}, label=label)

    end = datetime.datetime.now()
    msg = {}
    msg['timestamp'] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    msg['request'] = request.dict()
    msg['prediction'] = response.dict()
    diff = end-start    
    latency = diff.total_seconds() * 1000
    msg['latency'] = latency
    logger.info(msg)

    return response

@app.get("/")
def read_root():
    return {"Hello": "World"}
