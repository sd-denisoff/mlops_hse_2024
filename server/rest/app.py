from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from models import ModelManager


app = FastAPI()

model_manager = ModelManager("./saved_models")


class ModelSpec(BaseModel):
    type: str
    parameters: Dict[str, Any]


class TrainRequest(BaseModel):
    model_spec: ModelSpec
    features: List[Dict[str, float]]
    targets: List[float]


class PredictRequest(BaseModel):
    model_id: str
    features: List[Dict[str, float]]


@app.get("/models/")
async def list_models():
    return {model_class: model_class._get_param_names() for model_class in model_manager.model_classes.values()}


@app.post("/train/")
async def train_model(train_request: TrainRequest):
    model_type = train_request.model_spec.type
    params = train_request.model_spec.parameters
    features = train_request.features
    targets = train_request.targets
    try:
        model_id = model_manager.train_and_save_model(model_type, features, targets, params)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "success", "model_id": model_id}


@app.post("/predict/")
async def predict(request: PredictRequest):
    model_id = request.model_id
    features = request.features

    if model_id not in model_manager.list_models():
        raise HTTPException(status_code=404, detail='Not found model ID')

    try:
        model = model_manager.load_model(model_id)
        predictions = model.predict(features)
        return {"model_id": model_id, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    try:
        model_manager.delete_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"status": "success", "detail": "Model deleted successfully"}
