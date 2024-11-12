"""
Serve REST
"""

import uvicorn


def serve():
    """
    Run REST server
    """
    config = uvicorn.Config("app:app", port=8080, reload=True)
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    serve()
