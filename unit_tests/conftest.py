"""
Pytest fixtures
"""
import boto3
import pytest
from moto import mocqk_s3


@pytest.fixture(scope="session")
def mocked_s3():
    with mock_s3():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket_name = "test-bucket"
        s3.create_bucket(Bucket=bucket_name)
        yield {"s3_client": s3, "bucket_name": bucket_name}
