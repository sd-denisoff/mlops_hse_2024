"""
Serve gRPC
"""

from concurrent import futures

import grpc

from server.grpc.app import ModelService
from server.grpc.proto import model_service_pb2_grpc


def serve():
    """
    Run gRPC server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
