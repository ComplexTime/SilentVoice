import csv
import io
import random

from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict

from remotezip import RemoteZip
from tqdm import tqdm


URL = (
    "https://download.microsoft.com/download/"
    "b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/"
    "ASL_Citizen.zip"
)

ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "data" / "wlasl"
CITIZEN_DIR = OUTPUT / "citizen"
MANIFEST = OUTPUT / "manifest.csv"

MAX_PER_SIGN = 100

TARGETS = {
    "I LOVE YOU": "I_LOVE_YOU",
    "YES": "YES",
    "NO": "NO",
    "MORE": "MORE",
    "STOP": "STOP",
    "FINISH": "FINISH",

    # Additional negative examples
    "BOOK": "UNKNOWN",
    "COMPUTER": "UNKNOWN",
    "BLUE": "UNKNOWN",
    "SCHOOL": "UNKNOWN",
    "CHAIR": "UNKNOWN",
    "CAR": "UNKNOWN"
}

FIELDS = [
    "video_id",
    "label",
    "gloss",
    "signer_id",
    "path"
]


def normalize(text):
    return " ".join(
        text.upper()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def save_manifest(records):

    temporary = MANIFEST.with_suffix(".tmp")

    with temporary.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(records)

    temporary.replace(MANIFEST)


def main():

    CITIZEN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing = []

    if MANIFEST.exists():
        with MANIFEST.open(
            encoding="utf-8", newline=""
        ) as file:
            existing = list(csv.DictReader(file))

    existing_ids = {
        row["video_id"] for row in existing
    }

    counts = Counter(
        row["label"] for row in existing
    )

    print("Existing recordings:")
    print(counts)

    print("\nConnecting to ASL Citizen archive...")

    with RemoteZip(
        URL,
        support_suffix_range=False
    ) as archive:

        names = archive.namelist()

        print("Archive entries:", len(names))

        videos = {
            PurePosixPath(name).name: name
            for name in names
            if name.lower().endswith(".mp4")
        }

        # Keep Microsoft's official test split
        # untouched for future evaluation.

        csv_files = [
            name for name in names
            if PurePosixPath(name).name.lower()
            in ("train.csv", "val.csv")
        ]

        if not csv_files:
            raise RuntimeError(
                "Could not find dataset metadata."
            )

        candidates = defaultdict(list)
        seen = set(existing_ids)

        for csv_path in csv_files:

            content = archive.read(csv_path).decode(
                "utf-8-sig"
            )

            reader = csv.DictReader(
                io.StringIO(content)
            )

            print("Reading:", csv_path)
            print("Columns:", reader.fieldnames)

            for row in reader:

                row = {
                    k.strip().lower(): v
                    for k, v in row.items()
                    if k is not None
                }

                gloss = normalize(
                    row.get("gloss", "")
                )

                if gloss not in TARGETS:
                    continue

                label = TARGETS[gloss]

                filename = PurePosixPath(
                    row.get("video file", "")
                ).name

                signer = row.get(
                    "participant id", ""
                ).strip()

                if not signer:
                    continue

                if filename not in videos:
                    continue

                video_id = "citizen_" + filename

                if video_id in seen:
                    continue

                seen.add(video_id)

                candidates[label].append({
                    "video_id": video_id,
                    "label": label,
                    "gloss": gloss,
                    "signer_id": "citizen_" + signer,
                    "path": "citizen/" + filename,
                    "remote_path": videos[filename]
                })

        print("\nAvailable additional recordings:")

        for label, items in sorted(
            candidates.items()
        ):
            print(label, len(items))

        rng = random.Random(42)

        try:

            for label, items in sorted(
                candidates.items()
            ):

                rng.shuffle(items)

                remaining = max(
                    0,
                    MAX_PER_SIGN - counts[label]
                )

                print(
                    f"\nDownloading {label}: "
                    f"up to {remaining} additional videos"
                )

                for item in tqdm(items):

                    if counts[label] >= MAX_PER_SIGN:
                        break

                    destination = OUTPUT / item["path"]

                    try:

                        if (
                            not destination.exists()
                            or destination.stat().st_size == 0
                        ):

                            data = archive.read(
                                item["remote_path"]
                            )

                            if not data:
                                raise ValueError(
                                    "Empty video"
                                )

                            temporary = (
                                destination.with_suffix(".part")
                            )

                            temporary.write_bytes(data)
                            temporary.replace(destination)

                        record = {
                            key: item[key]
                            for key in FIELDS
                        }

                        existing.append(record)

                        counts[label] += 1

                    except Exception as error:

                        print(
                            "Download failed:",
                            item["video_id"],
                            error
                        )

        finally:

            save_manifest(existing)

    print("\n=== COMBINED DATASET ===")

    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")

    print("Total videos:", len(existing))
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()