"""
Tests
"""

import io


def test_save_model_to_minio(minio):
    """Test model saving"""
    minio_client = minio["minio_client"]
    bucket_name = minio["bucket_name"]
    model_key = "models/test_model.pkl"
    model_data = b"dummy model data"

    minio_client.put_object(bucket_name, model_key, data=io.BytesIO(model_data), length=len(model_data))

    response = minio_client.get_object(bucket_name, model_key)
    assert response.read() == model_data
