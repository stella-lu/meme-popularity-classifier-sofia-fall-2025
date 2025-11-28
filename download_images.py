import os
import json
import requests
from pathlib import Path

JSON_PATH = "reddit_meme_data.json"
OUTPUT_DIR = Path("image_data")
TIMEOUT = 10

def get_label(ups):
    if ups is None:
        return "low"  # fallback bucket
    
    if ups <= 2000:
        return "low"
    elif ups <= 20000:
        return "medium"
    else:
        return "high"

for lbl in ["low", "medium", "high"]:
    (OUTPUT_DIR / lbl).mkdir(parents=True, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

entries = data["_default"]

print(f"Found {len(entries)} meme entries.")
print("Starting image download...\n")

success_count = 0
fail_count = 0

for key, item in entries.items():
    media_url = item.get("media", None)
    ups = item.get("ups", None)
    meme_id = item.get("id", key)

    # skip anything without an image
    if not media_url or not isinstance(media_url, str):
        fail_count += 1
        continue

    # determine label
    label = get_label(ups)
    label_dir = OUTPUT_DIR / label

    # figure out file extension
    ext = os.path.splitext(media_url)[1]
    if ext == "":
        fail_count += 1
        continue

    filename = label_dir / f"{meme_id}{ext}"

    # skip if file already exists
    if filename.exists():
        continue

    try:
        resp = requests.get(media_url, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()

        with open(filename, "wb") as f_out:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f_out.write(chunk)

        success_count += 1

    except Exception as e:
        fail_count += 1
        continue

print("\nDownload completed.")
print("-------------------")
print(f"Successful downloads: {success_count}")
print(f"Failed downloads:     {fail_count}")
print("Images organized under image_data/low, image_data/medium, image_data/high")
