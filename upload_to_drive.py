import os
import base64
import pickle

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_credentials_from_env() -> Credentials:
    """
    Load OAuth2 Credentials from the GOOGLE_OAUTH_TOKEN env var.
    GOOGLE_OAUTH_TOKEN should be a base64-encoded token.pickle file.
    """
    token_b64 = os.environ.get("GOOGLE_OAUTH_TOKEN")
    if not token_b64:
        raise RuntimeError("GOOGLE_OAUTH_TOKEN not set")

    # Decode base64 back into the original pickled bytes
    token_bytes = base64.b64decode(token_b64)
    creds = pickle.loads(token_bytes)

    # Refresh if needed
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("Google credentials are invalid and cannot be refreshed")

    return creds


def upload_to_drive(local_path: str, drive_filename: str | None = None) -> str:
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("GOOGLE_DRIVE_FOLDER_ID not set")

    if drive_filename is None:
        drive_filename = os.path.basename(local_path)

    creds = get_credentials_from_env()
    service = build("drive", "v3", credentials=creds)

    # Pick a MIME type
    if local_path.lower().endswith(".csv"):
        mimetype = "text/csv"
    elif local_path.lower().endswith(".xlsx"):
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        mimetype = "application/octet-stream"

    file_metadata = {
        "name": drive_filename,
        "parents": [folder_id],
    }

    media = MediaFileUpload(
        local_path,
        mimetype=mimetype,
        resumable=False,
    )

    created = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )

    file_id = created.get("id")
    print(f"Uploaded to Drive: {drive_filename} (id={file_id})")
    return file_id
