"""
Tests
"""


def test_save_model_to_mocked_s3(mocked_s3):
    """Test model saving"""
    s3_client = mocked_s3["s3_client"]
    bucket_name = mocked_s3["bucket_name"]
    model_key = "models/test_model.pkl"
    model_data = b"dummy model data"

    s3_client.put_object(Bucket=bucket_name, Key=model_key, Body=model_data)

    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    assert response["Body"].read() == model_data
