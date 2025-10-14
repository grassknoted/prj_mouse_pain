#!/usr/bin/env python3
"""
organize_annotations_and_videos.py

Usage
-----
python organize_annotations_and_videos.py /path/to/REMY2
"""

import errno
import re
import shutil
import sys
from pathlib import Path

# ─── Regex for prediction CSVs ───────────────────────────────────────────────
# e.g. prediction_shortened_2024-05-04_…tracking.mp4_3.csv
CSV_PATTERN = re.compile(
    r"^prediction_(?P<video_name>shortened_.+?\.mp4)_(?P<trial>\d+)\.csv$"
)

# ─── xattr-related errno codes that need a lighter copy ──────────────────────
_XATTR_ERRNOS = {errno.EAGAIN, errno.EHOSTDOWN, errno.EOPNOTSUPP}

def safe_copy(src: Path, dst: Path) -> None:
    """Copy `src` → `dst`, falling back to `shutil.copy` if xattrs fail."""
    try:
        shutil.copy2(src, dst)          # full metadata copy
    except OSError as e:
        if e.errno in _XATTR_ERRNOS:
            shutil.copy(src, dst)       # retry without extended attributes
        else:
            raise

# ─── Main ────────────────────────────────────────────────────────────────────
def main(root: Path) -> None:
    # ----- Locate source trees ------------------------------------------------
    videos_src = root / "Videos"
    ann_src    = root / "Annotations"
    if not ann_src.exists():            # typo-tolerant fallback
        ann_src = root / "Annotataions"

    if not videos_src.exists() or not ann_src.exists():
        sys.exit("Error: could not find 'Videos/' and 'Annotations/' folders under "
                 f"{root}. Check the path and spelling.")

    # ----- Create flat output folders ----------------------------------------
    all_vids_dir  = root / "ALL_VIDEOS"
    all_anns_dir  = root / "ALL_ANNOTATIONS"
    all_vids_dir.mkdir(exist_ok=True)
    all_anns_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Scan once for every prediction CSV, bucket by its true video name
    # -------------------------------------------------------------------------
    annotations_by_video = {}
    for csv_path in ann_src.rglob("prediction_*.csv"):
        m = CSV_PATTERN.match(csv_path.name)
        if not m:
            continue
        video_name = m.group("video_name")
        annotations_by_video.setdefault(video_name, []).append(csv_path)

    # Counters for the final report
    trial_count   = 0
    unique_videos = set()

    # -------------------------------------------------------------------------
    # 2. Copy every shortened_*.mp4 first, then its trials
    # -------------------------------------------------------------------------
    for mp4 in videos_src.rglob("shortened_*.*"):
        dest = all_vids_dir / mp4.name
        if not dest.exists():
            safe_copy(mp4, dest)

        # copy all its annotation CSVs (if any)
        for csv_path in annotations_by_video.get(mp4.name, []):
            dest_csv = all_anns_dir / csv_path.name
            if not dest_csv.exists():
                safe_copy(csv_path, dest_csv)
            trial_count += 1

        if mp4.name in annotations_by_video:
            unique_videos.add(mp4.name)

    # -------------------------------------------------------------------------
    # 3. Flatten the *rest* of the files in Videos/ (DLC, etc.)
    #    Skip the MP4s we just handled.
    # -------------------------------------------------------------------------
    for other in videos_src.rglob("*"):
        if not other.is_file():
            continue
        if other.name.startswith("shortened_"):
            continue  # already copied above

        dest_other = all_vids_dir / other.name
        if not dest_other.exists():
            safe_copy(other, dest_other)

    # -------------------------------------------------------------------------
    # 4. Report
    # -------------------------------------------------------------------------
    print(f"✔ Copied {trial_count} trial CSVs into {all_anns_dir}.")
    print(f"✔ They cover {len(unique_videos)} unique videos.")
    print(f"✔ All videos & DLC files flattened into {all_vids_dir}.")

# ─── Entry-point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python organize_annotations_and_videos.py <path/to/REMY2>")
        sys.exit(1)

    root_dir = Path(sys.argv[1]).expanduser().resolve()
    if not root_dir.exists():
        sys.exit(f"Error: {root_dir} does not exist.")

    main(root_dir)
#!/usr/bin/env python3
"""
organize_annotations_and_videos.py

Usage
-----
python organize_annotations_and_videos.py /path/to/REMY2
"""

import errno
import re
import shutil
import sys
from pathlib import Path

# ─── Regex for prediction CSVs ───────────────────────────────────────────────
# e.g. prediction_shortened_2024-05-04_…tracking.mp4_3.csv
CSV_PATTERN = re.compile(
    r"^prediction_(?P<video_name>shortened_.+?\.mp4)_(?P<trial>\d+)\.csv$"
)

# ─── xattr-related errno codes that need a lighter copy ──────────────────────
_XATTR_ERRNOS = {errno.EAGAIN, errno.EHOSTDOWN, errno.EOPNOTSUPP}

def safe_copy(src: Path, dst: Path) -> None:
    """Copy `src` → `dst`, falling back to `shutil.copy` if xattrs fail."""
    try:
        shutil.copy2(src, dst)          # full metadata copy
    except OSError as e:
        if e.errno in _XATTR_ERRNOS:
            shutil.copy(src, dst)       # retry without extended attributes
        else:
            raise

# ─── Main ────────────────────────────────────────────────────────────────────
def main(root: Path) -> None:
    # ----- Locate source trees ------------------------------------------------
    videos_src = root / "Videos"
    ann_src    = root / "Annotations"
    if not ann_src.exists():            # typo-tolerant fallback
        ann_src = root / "Annotataions"

    if not videos_src.exists() or not ann_src.exists():
        sys.exit("Error: could not find 'Videos/' and 'Annotations/' folders under "
                 f"{root}. Check the path and spelling.")

    # ----- Create flat output folders ----------------------------------------
    all_vids_dir  = root / "ALL_VIDEOS"
    all_anns_dir  = root / "ALL_ANNOTATIONS"
    all_vids_dir.mkdir(exist_ok=True)
    all_anns_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Scan once for every prediction CSV, bucket by its true video name
    # -------------------------------------------------------------------------
    annotations_by_video = {}
    for csv_path in ann_src.rglob("prediction_*.csv"):
        m = CSV_PATTERN.match(csv_path.name)
        if not m:
            continue
        video_name = m.group("video_name")
        annotations_by_video.setdefault(video_name, []).append(csv_path)

    # Counters for the final report
    trial_count   = 0
    unique_videos = set()

    # -------------------------------------------------------------------------
    # 2. Copy every shortened_*.mp4 first, then its trials
    # -------------------------------------------------------------------------
    for mp4 in videos_src.rglob("shortened_*.*"):
        dest = all_vids_dir / mp4.name
        if not dest.exists():
            safe_copy(mp4, dest)

        # copy all its annotation CSVs (if any)
        for csv_path in annotations_by_video.get(mp4.name, []):
            dest_csv = all_anns_dir / csv_path.name
            if not dest_csv.exists():
                safe_copy(csv_path, dest_csv)
            trial_count += 1

        if mp4.name in annotations_by_video:
            unique_videos.add(mp4.name)

    # -------------------------------------------------------------------------
    # 3. Flatten the *rest* of the files in Videos/ (DLC, etc.)
    #    Skip the MP4s we just handled.
    # -------------------------------------------------------------------------
    for other in videos_src.rglob("*"):
        if not other.is_file():
            continue
        if other.name.startswith("shortened_"):
            continue  # already copied above

        dest_other = all_vids_dir / other.name
        if not dest_other.exists():
            safe_copy(other, dest_other)

    # -------------------------------------------------------------------------
    # 4. Report
    # -------------------------------------------------------------------------
    print(f"✔ Copied {trial_count} trial CSVs into {all_anns_dir}.")
    print(f"✔ They cover {len(unique_videos)} unique videos.")
    print(f"✔ All videos & DLC files flattened into {all_vids_dir}.")

# ─── Entry-point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python organize_annotations_and_videos.py <path/to/REMY2>")
        sys.exit(1)

    root_dir = Path(sys.argv[1]).expanduser().resolve()
    if not root_dir.exists():
        sys.exit(f"Error: {root_dir} does not exist.")

    main(root_dir)
