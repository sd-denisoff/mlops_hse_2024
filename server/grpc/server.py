import grpc
from concurrent import futures
import modelapi_pb2
import modelapi_pb2_grpc


class ModelService(modelapi_pb2_grpc.ModelServiceServicer):
    def ListModels(self, request, context):
        return modelapi_pb2.ModelList(
            models={
                "linear": modelapi_pb2.ModelSpec(parameters=["coef_", "intercept_"]),
                "boosting": modelapi_pb2.ModelSpec(parameters=["learning_rate", "depth", "iterations"]),
            }
        )

    def TrainModel(self, request, context):
        model_type = request.type
        params = request.parameters
        features = [dict(f.features) for f in request.features]
        targets = list(request.targets)

        model_id = str(uuid.uuid4())
        try:
            model_manager.train_and_save_model(model_type, model_id, features, targets, params)
            return modelapi_pb2.TrainResponse(status="success", model_id=model_id)
        except Exception as e:
            context.abort(grpc.StatusCode.ABORTED, str(e))

    # Реализация других методов аналогично.


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    modelapi_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
