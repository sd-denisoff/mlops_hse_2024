"""
gRPC server implementation
"""

# pylint: disable=no-member, broad-exception-caught

from models.model_manager import MODEL_MANAGER
from server.grpc.proto import model_service_pb2, model_service_pb2_grpc


class ModelService(model_service_pb2_grpc.ModelServiceServicer):
    """
    Service methods
    """

    def list_models(self, request, context):
        """
        list_models method implementation
        """
        return model_service_pb2.ModelList(
            models={
                "linear": model_service_pb2.ModelSpec(
                    parameters=["coef_", "intercept_"],
                ),
                "boosting": model_service_pb2.ModelSpec(
                    parameters=["learning_rate", "depth", "iterations"],
                ),
            }
        )

    def train_model(self, request, context):
        """
        train_model method implementation
        """
        model_type = request.type
        params = request.parameters
        features = [dict(f.features) for f in request.features]
        targets = list(request.targets)

        try:
            model_id = MODEL_MANAGER.train_and_save_model(
                model_type=model_type,
                X_train=features,
                y_train=targets,
                model_params=params,
            )
            return model_service_pb2.TrainResponse(status="success", model_id=model_id)
        except Exception as exc:
            return model_service_pb2.TrainResponse(status=str(exc), model_id=None)
