"""
Tests
"""


def test_save_model_to_mocked_s3(mocked_s3):
    """Test model saving"""
    # retrieve s3 connection
    s3_client = mocked_s3["s3_client"]
    bucket_name = mocked_s3["bucket_name"]

    # prepare model
    model_key = "models/test_model.pkl"
    pickled_model_data = b"dummy ML model data for saving test"

    # put object
    s3_client.put_object(Bucket=bucket_name, Key=model_key, Body=pickled_model_data)

    # check saved object
    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    assert response["Body"].read() == pickled_model_data
