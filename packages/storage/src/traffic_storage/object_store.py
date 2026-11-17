"""MinIO/S3 object storage adapter."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from traffic_config.settings import StorageSettings, get_settings
from traffic_observability.logging import get_logger


logger = get_logger(__name__)


class MinIOAdapter:
    """Adapter for MinIO/S3 object storage."""

    def __init__(self, settings: StorageSettings | None = None):
        self._settings = settings or get_settings().storage
        self._client: Minio | None = None

    def connect(self) -> None:
        """Initialize MinIO client."""
        self._client = Minio(
            self._settings.s3_endpoint,
            access_key=self._settings.s3_access_key,
            secret_key=self._settings.s3_secret_key,
            secure=self._settings.s3_secure,
            region=self._settings.s3_region,
        )

        # Ensure bucket exists
        if not self._client.bucket_exists(self._settings.s3_bucket):
            self._client.make_bucket(self._settings.s3_bucket)
            logger.info("Created bucket", bucket=self._settings.s3_bucket)

        logger.info("MinIO connection established", endpoint=self._settings.s3_endpoint)

    def close(self) -> None:
        """Close MinIO client (no-op for MinIO)."""
        self._client = None
        logger.info("MinIO connection closed")

    def upload_file(
        self,
        object_name: str,
        file_path: str | Path,
        content_type: str | None = None,
    ) -> str:
        """Upload a file to object storage.

        Args:
            object_name: Object name/key in bucket
            file_path: Local file path
            content_type: MIME content type

        Returns:
            Object URL
        """
        if not self._client:
            self.connect()

        file_path = Path(file_path)
        self._client.fput_object(
            self._settings.s3_bucket,
            object_name,
            str(file_path),
            content_type=content_type,
        )

        url = f"s3://{self._settings.s3_bucket}/{object_name}"
        logger.debug("File uploaded", object=object_name, url=url)
        return url

    def upload_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> str:
        """Upload bytes to object storage."""
        if not self._client:
            self.connect()

        from io import BytesIO

        self._client.put_object(
            self._settings.s3_bucket,
            object_name,
            BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

        return f"s3://{self._settings.s3_bucket}/{object_name}"

    def download_file(self, object_name: str, file_path: str | Path) -> None:
        """Download a file from object storage."""
        if not self._client:
            self.connect()

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self._client.fget_object(
            self._settings.s3_bucket,
            object_name,
            str(file_path),
        )

        logger.debug("File downloaded", object=object_name, path=file_path)

    def download_bytes(self, object_name: str) -> bytes:
        """Download object as bytes."""
        if not self._client:
            self.connect()

        response = self._client.get_object(self._settings.s3_bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete_object(self, object_name: str) -> None:
        """Delete an object."""
        if not self._client:
            self.connect()

        self._client.remove_object(self._settings.s3_bucket, object_name)
        logger.debug("Object deleted", object=object_name)

    def list_objects(self, prefix: str = "", recursive: bool = True) -> list[str]:
        """List objects in bucket."""
        if not self._client:
            self.connect()

        objects = self._client.list_objects(
            self._settings.s3_bucket,
            prefix=prefix,
            recursive=recursive,
        )
        return [obj.object_name for obj in objects]

    def object_exists(self, object_name: str) -> bool:
        """Check if object exists."""
        if not self._client:
            self.connect()

        try:
            self._client.stat_object(self._settings.s3_bucket, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    def get_presigned_url(
        self,
        object_name: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Generate presigned URL for object access."""
        if not self._client:
            self.connect()

        from datetime import timedelta

        url = self._client.presigned_get_object(
            self._settings.s3_bucket,
            object_name,
            expires=timedelta(seconds=expires_seconds),
        )
        return url