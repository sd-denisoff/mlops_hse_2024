"""
Pytest fixtures
"""
import pytest
from minio import Minio


@pytest.fixture(scope="session")
def real_minio():
    # Конфигурация реального MinIO
    minio_client = Minio(
        "localhost:9000",  # Укажите хост MinIO
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,  # Если используется HTTP
    )
    bucket_name = "real-bucket"

    # Создаем bucket, если его нет
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    yield {"minio_client": minio_client, "bucket_name": bucket_name}
