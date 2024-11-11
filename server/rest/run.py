"""
Serve REST
"""
import uvicorn 


def serve():
    """
    Run REST server
    """
    config = uvicorn.Config("app:app", port=8000)
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    serve()
