import csv
import random
from pathlib import Path

# === Configuration ===
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
random.seed(42)

# Control settings
FRAME_INTERVAL = 1      # how far source frames are from target
VAL_RATIO = 0.1         # 10% of scenes for validation
TEST_RATIO = 0.05       # 5% of scenes for test


def get_intrinsics_path(date_dir: Path):
    """Return calibration file path for a given date directory."""
    calib_dir = date_dir
    calib_file = calib_dir / "calib_cam_to_cam.txt"
    return calib_file if calib_file.exists() else None


def collect_sequences():
    """Collect all drive sequences with available image and optional depth data."""
    sequences = []

    for date_dir in sorted(DATA_DIR.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("2011_"):
            continue

        calib_path = get_intrinsics_path(date_dir)
        if calib_path is None:
            print(f"[WARN] No calibration found for {date_dir.name}")
            continue

        for drive_dir in date_dir.glob("*_drive_*_sync"):
            rgb_dir = drive_dir / "image_02" / "data"
            depth_dir = drive_dir / "velodyne_points" / "data"

            if not rgb_dir.exists():
                continue

            rgb_files = sorted(rgb_dir.glob("*.png"))
            if len(rgb_files) < 3:
                continue  # need at least 3 frames for self-supervised triplets

            sequences.append({
                "date": date_dir.name,
                "drive": drive_dir.name,
                "rgb_files": rgb_files,
                "depth_dir": depth_dir if depth_dir.exists() else None,
                "intrinsics": calib_path
            })

    return sequences


def make_triplets(sequence, frame_interval=1):
    """Generate (target, src1, src2, intrinsics, [optional depth]) triplets."""
    triplets = []
    rgb_files = sequence["rgb_files"]
    calib = sequence["intrinsics"]
    depth_dir = sequence["depth_dir"]

    for i in range(frame_interval, len(rgb_files) - frame_interval):
        target = rgb_files[i]
        src1 = rgb_files[i - frame_interval]
        src2 = rgb_files[i + frame_interval]

        depth_path = None
        if depth_dir and (depth_dir / target.name).exists():
            depth_path = (depth_dir / target.name)

        triplet = {
            "target": target,
            "src1": src1,
            "src2": src2,
            "intrinsics": calib,
            "depth": depth_path
        }
        triplets.append(triplet)

    return triplets


def scene_split(sequences, val_ratio=0.1, test_ratio=0.05):
    """Split dataset by scene (no overlapping drives between splits)."""
    random.shuffle(sequences)
    n_total = len(sequences)
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))

    val_scenes = sequences[:n_val]
    test_scenes = sequences[n_val:n_val + n_test]
    train_scenes = sequences[n_val + n_test:]

    def expand(scene_list):
        samples = []
        for seq in scene_list:
            samples += make_triplets(seq, FRAME_INTERVAL)
        return samples

    return expand(train_scenes), expand(val_scenes), expand(test_scenes)


def write_csv(samples, filename):
    """Write samples to CSV."""
    csv_path = SPLITS_DIR / filename
    header = [
        "target_filepath",
        "src1_filepath",
        "src2_filepath",
        "camera_intrinsics_path",
        "depth_filepath"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for s in samples:
            writer.writerow([
                s["target"],
                s["src1"],
                s["src2"],
                s["intrinsics"],
                s["depth"] or ""
            ])

    print(f"[OK] {filename}: {len(samples)} samples written → {csv_path}")


def main():
    print("[INFO] Collecting KITTI sequences...")
    sequences = collect_sequences()
    print(f"[INFO] Found {len(sequences)} valid sequences")

    train_samples, val_samples, test_samples = scene_split(sequences, VAL_RATIO, TEST_RATIO)

    print(f"[INFO] Split sizes — Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    write_csv(train_samples, "train_selfsup.csv")
    write_csv(val_samples, "val_selfsup.csv")
    write_csv(test_samples, "test_selfsup.csv")


if __name__ == "__main__":
    main()
