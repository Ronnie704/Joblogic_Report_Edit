import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def upload_to_drive(local_path: str, drive_filename: str | None = None) -> str:
  folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
  if not folder_id:
    raise RuntimeError("GOOGLE_DRIVE_FOLDER_ID not set")

  if drive_filename is None:
    drive_filename = os.path.basename(local_path)
    
  creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
  )

  service = build("drive", "v3", credentials=creds)

  if local_path.lower().endswith(".csv"):
    mimetype = "text/csv"
  else:
    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

  file_metadata = {
    "name": drive_filename,
    "parents": [folder_id],
  }

  media = MediaFileUpload(
    local_path,
    mimetype=mimetype
    resumable=True,
  )

  created = (
    service.files()
    .create(body=file_metadata, media_body=media, fields="id")
    .execute()
  )

  file_id = created.get("id")
  print(f"Upload to Drive: {drive_filename} (id={file_id})")
  return file_id
