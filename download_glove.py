"""
GloVe Embeddings Downloader
============================
Downloads the GloVe 6B 100-dimensional pre-trained word vectors from Stanford
and extracts only the file needed by train.py.

Source : https://nlp.stanford.edu/projects/glove/
File   : glove.6B.zip  (~822 MB download)
Extracts: glove/glove.6B.100d.txt  (~347 MB)

Usage:
    python download_glove.py
"""

import os
import zipfile
import urllib.request

GLOVE_URL    = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR    = "glove"
ZIP_PATH     = os.path.join(GLOVE_DIR, "glove.6B.zip")
TARGET_FILE  = "glove.6B.100d.txt"          # only this file is extracted
TARGET_PATH  = os.path.join(GLOVE_DIR, TARGET_FILE)


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    bar = "█" * int(pct / 2)
    print(f"\r  [{bar:<50}] {pct:.1f}%  ({downloaded // 1_048_576} MB)", end="", flush=True)


def main():
    os.makedirs(GLOVE_DIR, exist_ok=True)

    if os.path.isfile(TARGET_PATH):
        print(f"✅  GloVe already downloaded: {TARGET_PATH}")
        print("   You can now run  python train.py  to use it.")
        return

    print(f"Downloading GloVe 6B embeddings from Stanford (~822 MB)...")
    print(f"  → {GLOVE_URL}\n")

    urllib.request.urlretrieve(GLOVE_URL, ZIP_PATH, reporthook=_progress)
    print("\n\nDownload complete. Extracting glove.6B.100d.txt ...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extract(TARGET_FILE, GLOVE_DIR)

    print(f"Extracted → {TARGET_PATH}")

    # Remove the zip to save space (the zip contains 4 files; only 100d needed)
    os.remove(ZIP_PATH)
    print("Removed zip file to save disk space.")
    print(f"\n✅  Done! Run  python train.py  to retrain with GloVe embeddings.")


if __name__ == "__main__":
    main()
