until mc ls minio > /dev/null 2>&1; do
    sleep 0.5
done

mc mb minio/post-bucket