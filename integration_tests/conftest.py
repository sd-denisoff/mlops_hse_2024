"""
Pytest fixtures
"""
import pytest
from minio import Minio


@pytest.fixture(scope="session")
def minio():
    minio_client = Minio(
        "localhost:9000",
        access_key="user",  # todo: retrieve from environment
        secret_key="password",  # todo: retrieve from environment
        secure=False,
    )
    bucket_name = "models-bucket"

    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    yield {"minio_client": minio_client, "bucket_name": bucket_name}
