"""
Tests
"""


def test_save_model_to_real_minio(real_minio):
    """Test model saving"""
    minio_client = real_minio["minio_client"]
    bucket_name = real_minio["bucket_name"]
    model_key = "models/test_model.pkl"
    model_data = b"dummy model data"

    # Загружаем модель в реальный MinIO
    minio_client.put_object(bucket_name, model_key, data=model_data, length=len(model_data))

    # Проверяем, что файл существует
    response = minio_client.get_object(bucket_name, model_key)
    assert response.read() == model_data
