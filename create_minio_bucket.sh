#!/bin/bash

until mc ls minio > /dev/null 2>&1; do
    sleep 0.5
done

mc alias set myminio http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}
mc mb myminio/${MINIO_BUCKET}
mc ls myminio
