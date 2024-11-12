"""
REST server implementation
"""

from typing import Any

from fastapi import FastAPI, HTTPException
from pandas import DataFrame
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


@app.get("/status")
async def get_status():
    """
    Возвращает статус сервиса
    """
    return {"status": "online"}


@app.get("/models")
async def list_models():
    """
    list_models method implementation
    """
    return {
        model.__name__: model.get_param_names()
        for model in MODEL_MANAGER.model_classes
    }


@app.get("/trained_models")
async def list_trained_models():
    """
    list_trained_models method implementation
    """
    return {"trained_models": MODEL_MANAGER.list_models()}


@app.post("/train")
async def train_model(train_request: TrainRequest):
    """
    train_model method implementation
    """
    model_type = train_request.model_spec.type
    params = train_request.model_spec.parameters
    features = DataFrame(train_request.features)
    targets = train_request.targets

    try:
        model_id = MODEL_MANAGER.train_and_save_model(
            model_type=model_type,
            X_train=features,
            y_train=targets,
            model_params=params,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "success", "model_id": model_id}


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    predict method implementation
    """
    model_id = request.model_id
    features = DataFrame(request.features)

    if model_id not in MODEL_MANAGER.list_models():
        raise HTTPException(status_code=404, detail="Not found model ID")

    try:
        model = MODEL_MANAGER.load_model(model_id)
        predictions = list(model.predict(features))
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
