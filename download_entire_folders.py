from huggingface_hub import snapshot_download
import os


REPO_ID = "hungphongtrn/vqav2_extracted_features"
snapshot_download(repo_id=REPO_ID, max_workers=os.cpu_count())