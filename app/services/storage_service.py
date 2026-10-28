"""
app/services/storage_service.py
Supabase Storage helpers for generated CTC PDFs and citizen profile pictures.

Files are kept in private buckets and only the backend touches them, using the
server-side secret key. The database stores a reference of the form
"supabase://<bucket>/<object path>" in place of a local path or inline image.

When SUPABASE_URL / SUPABASE_SECRET_KEY are not set, storage is disabled and
callers fall back to the original behaviour (local PDFs, inline base64 images).
"""
import base64
import binascii
import logging
import os
import re
import uuid
from typing import Optional, Tuple

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

STORAGE_PREFIX = "supabase://"
_DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
_IMAGE_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


class StorageError(RuntimeError):
    pass


def is_enabled() -> bool:
    return bool(settings.SUPABASE_URL and settings.SUPABASE_SECRET_KEY)


def is_storage_ref(value: Optional[str]) -> bool:
    return bool(value) and value.startswith(STORAGE_PREFIX)


def make_ref(bucket: str, path: str) -> str:
    return f"{STORAGE_PREFIX}{bucket}/{path}"


def parse_ref(ref: str) -> Tuple[str, str]:
    bucket, _, path = ref[len(STORAGE_PREFIX):].partition("/")
    if not bucket or not path:
        raise StorageError(f"Malformed storage reference: {ref!r}")
    return bucket, path


def _headers() -> dict:
    key = settings.SUPABASE_SECRET_KEY
    headers = {"apikey": key}
    # Legacy service_role keys are JWTs and also go in Authorization.
    # New sb_secret_ keys must NOT be sent as a Bearer token.
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _object_url(bucket: str, path: str) -> str:
    return f"{settings.SUPABASE_URL}/storage/v1/object/{bucket}/{path}"


def upload_bytes(bucket: str, path: str, data: bytes, content_type: str) -> str:
    """Upload (or overwrite) an object and return its storage reference."""
    headers = {**_headers(), "Content-Type": content_type, "x-upsert": "true"}
    response = httpx.post(_object_url(bucket, path), content=data, headers=headers, timeout=60)
    if response.status_code >= 400:
        raise StorageError(f"Upload to {bucket}/{path} failed: {response.status_code} {response.text[:200]}")
    return make_ref(bucket, path)


def download_bytes(ref: str) -> Tuple[bytes, str]:
    """Return (content, content_type) for a storage reference."""
    bucket, path = parse_ref(ref)
    response = httpx.get(_object_url(bucket, path), headers=_headers(), timeout=60)
    # Storage answers a missing object with 400 + "not_found" (older) or 404.
    if response.status_code == 404 or (response.status_code == 400 and "not_found" in response.text):
        raise FileNotFoundError(ref)
    if response.status_code >= 400:
        raise StorageError(f"Download of {ref} failed: {response.status_code} {response.text[:200]}")
    return response.content, response.headers.get("content-type", "application/octet-stream")


def delete_object_strict(ref: str) -> None:
    """Delete an object and raise StorageError when the storage service did not confirm it (plan 003: the
    admin's "Delete evaluation audio now" must not report a clip gone while it is still stored)."""
    bucket, path = parse_ref(ref)
    response = httpx.request(
        "DELETE",
        f"{settings.SUPABASE_URL}/storage/v1/object/{bucket}",
        json={"prefixes": [path]},
        headers=_headers(),
        timeout=30,
    )
    if response.status_code >= 400:
        raise StorageError(f"Could not delete {ref}: {response.status_code}")


def delete_object(ref: str) -> None:
    """Best-effort delete; logs instead of raising so DB changes are never blocked."""
    try:
        bucket, path = parse_ref(ref)
        response = httpx.request(
            "DELETE",
            f"{settings.SUPABASE_URL}/storage/v1/object/{bucket}",
            json={"prefixes": [path]},
            headers=_headers(),
            timeout=30,
        )
        if response.status_code >= 400:
            logger.warning("Could not delete %s: %s", ref, response.status_code)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete %s: %s", ref, exc)


# ─── CTC PDFs ────────────────────────────────────────────────────────────────

def store_pdf(local_path: str) -> str:
    """
    Upload a generated PDF and return the value to save in ctc_issuance.pdf_path.
    Returns the local path unchanged when storage is disabled.
    """
    if not is_enabled():
        return local_path
    with open(local_path, "rb") as fh:
        data = fh.read()
    return upload_bytes(settings.SUPABASE_PDF_BUCKET, os.path.basename(local_path), data, "application/pdf")


# ─── Profile pictures ────────────────────────────────────────────────────────

def decode_image_data_url(value: Optional[str]) -> Optional[Tuple[bytes, str]]:
    """Return (bytes, content_type) for an image data URL, or None if it is not one."""
    match = _DATA_URL_RE.match(value.strip()) if value else None
    if not match:
        return None
    content_type = match.group(1).lower()
    if content_type == "image/jpg":
        content_type = "image/jpeg"
    if content_type not in _IMAGE_EXTENSIONS:
        raise StorageError(f"Unsupported image type: {content_type}")
    try:
        return base64.b64decode(match.group(2), validate=False), content_type
    except (binascii.Error, ValueError) as exc:
        raise StorageError("Profile picture is not valid base64.") from exc


def store_profile_picture(
    user_id: int, value: Optional[str], object_name: Optional[str] = None
) -> Optional[str]:
    """
    Turn an incoming base64 data URL into a storage reference.
    Values that are not data URLs (or when storage is disabled) are returned unchanged.
    """
    if not value or not is_enabled():
        return value
    decoded = decode_image_data_url(value)
    if decoded is None:
        return value
    data, content_type = decoded
    name = object_name or uuid.uuid4().hex
    path = f"citizens/{user_id}/{name}.{_IMAGE_EXTENSIONS[content_type]}"
    return upload_bytes(settings.SUPABASE_PROFILE_BUCKET, path, data, content_type)


def profile_picture_for_client(value: Optional[str]) -> Optional[str]:
    """
    Return the profile picture the way clients have always received it: a
    base64 data URL. Storage references are fetched and re-encoded, so the
    mobile app needs no changes and no public or expiring URLs are handed out.
    """
    if not is_storage_ref(value):
        return value
    if not is_enabled():
        return None
    try:
        data, content_type = download_bytes(value)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Profile picture %s unavailable: %s", value, exc)
        return None
    return f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"
