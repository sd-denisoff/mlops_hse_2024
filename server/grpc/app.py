"""
gRPC server implementation
"""

import uuid

import grpc

import model_service_pb2
import model_service_pb2_grpc
from models import MODEL_MANAGER


class ModelService(model_service_pb2_grpc.ModelServiceServicer):
    def ListModels(self, request, context):
        return model_service_pb2.ModelList(
            models={
                "linear": model_service_pb2.ModelSpec(parameters=["coef_", "intercept_"]),
                "boosting": model_service_pb2.ModelSpec(parameters=["learning_rate", "depth", "iterations"]),
            }
        )

    def TrainModel(self, request, context):
        model_type = request.type
        params = request.parameters
        features = [dict(f.features) for f in request.features]
        targets = list(request.targets)

        model_id = str(uuid.uuid4())
        try:
            MODEL_MANAGER.train_and_save_model(model_type, model_id, features, targets, params)
            return model_service_pb2.TrainResponse(status="success", model_id=model_id)
        except Exception as e:
            context.abort(grpc.StatusCode.ABORTED, str(e))

    # todo: Реализация других методов аналогично
