"""
REST server implementation
"""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.model_manager import MODEL_MANAGER

app = FastAPI()


class ModelSpec(BaseModel):
    """ModelSpec model"""
    type: str
    parameters: dict[str, Any]


class TrainRequest(BaseModel):
    """TrainRequest model"""
    model_spec: ModelSpec
    features: list[dict[str, float]]
    targets: list[float]


class PredictRequest(BaseModel):
    """PredictRequest model"""
    model_id: str
    features: list[dict[str, float]]


@app.get("/models/")
async def list_models():
    """
    list_models method implementation
    """
    return {
        model_class.__class__.__name__: model_class.get_param_names()
        for model_class in MODEL_MANAGER.model_classes.values()
    }


@app.post("/train/")
async def train_model(train_request: TrainRequest):
    """
    train_model method implementation
    """
    model_type = train_request.model_spec.type
    params = train_request.model_spec.parameters
    features = train_request.features
    targets = train_request.targets
    try:
        model_id = MODEL_MANAGER.train_and_save_model(model_type, features, targets, params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "success", "model_id": model_id}


@app.post("/predict/")
async def predict(request: PredictRequest):
    """
    predict method implementation
    """
    model_id = request.model_id
    features = request.features

    if model_id not in MODEL_MANAGER.list_models():
        raise HTTPException(status_code=404, detail='Not found model ID')

    try:
        model = MODEL_MANAGER.load_model(model_id)
        predictions = model.predict(features)
        return {"model_id": model_id, "predictions": predictions}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    delete_model method implementation
    """
    try:
        MODEL_MANAGER.delete_model(model_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"status": "success", "detail": "Model deleted successfully"}
