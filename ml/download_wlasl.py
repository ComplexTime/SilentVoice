import csv
import random
from pathlib import Path
from collections import Counter

import requests
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


REPO_ID = "Voxel51/WLASL"

METADATA_URL = (
    "https://raw.githubusercontent.com/"
    "dxli94/WLASL/master/start_kit/"
    "WLASL_v0.3.json"
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "wlasl"

MAX_PER_SIGN = 60

TARGETS = {
    "i love you": "I_LOVE_YOU",
    "ily": "I_LOVE_YOU",
    "yes": "YES",
    "no": "NO",
    "more": "MORE",
    "stop": "STOP",
    "finish": "FINISH",
    "book": "UNKNOWN",
    "computer": "UNKNOWN",
    "blue": "UNKNOWN",
    "school": "UNKNOWN"
}


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Downloading WLASL metadata...")

    response = requests.get(
        METADATA_URL,
        timeout=120
    )
    response.raise_for_status()

    metadata = response.json()

    print("Getting available video files...")

    files = HfApi().list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset"
    )

    video_paths = {
        Path(path).stem: path
        for path in files
        if path.endswith(".mp4")
    }

    candidates = []

    for entry in metadata:

        gloss = entry["gloss"].lower().strip()
        gloss = gloss.replace("_", " ")
        gloss = gloss.replace("-", " ")

        if gloss not in TARGETS:
            continue

        label = TARGETS[gloss]

        for instance in entry["instances"]:

            video_id = str(instance["video_id"])
            signer_id = instance.get("signer_id")

            if signer_id is None:
                continue

            if video_id not in video_paths:
                continue

            candidates.append({
                "video_id": video_id,
                "label": label,
                "gloss": gloss,
                "signer_id": str(signer_id),
                "remote_path": video_paths[video_id]
            })

    print("Available recordings:")
    print(Counter(c["label"] for c in candidates))

    random.Random(42).shuffle(candidates)

    counts = Counter()
    manifest = []

    for item in tqdm(candidates):

        label = item["label"]

        if counts[label] >= MAX_PER_SIGN:
            continue

        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=item["remote_path"],
                local_dir=str(OUTPUT)
            )

        except Exception as error:
            print("Download failed:", error)
            continue

        manifest.append({
            "video_id": item["video_id"],
            "label": label,
            "gloss": item["gloss"],
            "signer_id": item["signer_id"],
            "path": str(
                Path(path).relative_to(OUTPUT)
            )
        })

        counts[label] += 1

    manifest_path = OUTPUT / "manifest.csv"

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "video_id",
                "label",
                "gloss",
                "signer_id",
                "path"
            ]
        )

        writer.writeheader()
        writer.writerows(manifest)

    print("\nDownloaded recordings:")

    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")

    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()