"""
Serve REST
"""
import uvicorn 


def serve():
    """
    Run REST server
    """
    config = uvicorn.Config("app:app", port=8080)
    server = uvicorn.Server(config)
    
    try:  
        server.run()  
    finally:  
        server.stop()

if __name__ == "__main__":
    serve()
