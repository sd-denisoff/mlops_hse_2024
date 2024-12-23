# #!/bin/bash
mlflow server \
--host "0.0.0.0" \
--port ${MLFLOW_PORT} \
--backend-store-uri ${MLFLOW_BACKEND_STORE_URI} 
