"""
Pytest fixtures
"""

import os

import pytest
from dotenv import load_dotenv
from minio import Minio

load_dotenv()  # загрузка окружения из .env


@pytest.fixture(scope="session")
def minio() -> dict:
    """
    Create MinIO client
    """
    minio_port = os.getenv("MINIO_PORT", "9000")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")

    minio_client = Minio(
        f"localhost:{minio_port}",
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )

    bucket_name = "models-test-bucket"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    yield {"minio_client": minio_client, "bucket_name": bucket_name}
