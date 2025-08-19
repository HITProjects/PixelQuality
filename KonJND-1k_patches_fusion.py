import os
import json
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths ---
source_image_path = "KonJND-1k/source_image"
jpeg_image_path = "KonJND-1k/jpeg"
bpg_image_path = "KonJND-1k/bpg"
patch_image_path = "KonJND-1k/patches"

os.makedirs(patch_image_path, exist_ok=True)

# --- Read metadata CSV ---
metadata_file = pd.read_csv("KonJND-1k/subjective_ratings.csv")

all_patch_metadata = []
all_psnr_values = []

PATCH_SIZE = 20
CENTER_SIZE = 12
OFFSET = (PATCH_SIZE - CENTER_SIZE) // 2

for idx, row in metadata_file.iterrows():
    img_id = row["image_id"]
    img_id_base = os.path.splitext(img_id)[0]
    comp_type = row["Compression type"]
    ratings = json.loads(row["ratings"])

    # --- Build grade table for compression levels ---
    max_rating_value = 100 if comp_type == "JPEG" else 53
    grades = np.zeros(max_rating_value + 1, dtype=float)
    for r in ratings:
        grades[r + 1 :] += 1

    # Normalize grades to [0,1] severity
    if grades.max() > 0:
        grades /= grades.max()

    # --- Load source image ---
    source_image_file = os.path.join(source_image_path, img_id)
    source_image = Image.open(source_image_file).convert("RGB")

    # --- Choose one patch location ---
    max_x = source_image.width - PATCH_SIZE
    max_y = source_image.height - PATCH_SIZE
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    patch_region = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)

    # --- Sample a random target severity ---
    s = random.uniform(0, 1)

    # Find compression level whose grade is closest to s
    level_idx = np.argmin(np.abs(grades - s))

    # Compose distorted image filename
    if comp_type == "JPEG":
        distorted_filename = f"{img_id_base}_JPEG_0{level_idx:02d}.jpg"
        distorted_image_file = os.path.join(jpeg_image_path, distorted_filename)
    else:
        distorted_filename = f"{img_id_base}_BPG_0{level_idx:02d}.png"
        distorted_image_file = os.path.join(bpg_image_path, distorted_filename)

    if not os.path.exists(distorted_image_file):
        continue  # skip if distorted image is missing

    distorted_image = Image.open(distorted_image_file).convert("RGB")

    # --- Extract patches ---
    clean_patch = source_image.crop(patch_region)
    distorted_patch = distorted_image.crop(patch_region)

    # Fuse: distorted center pasted into clean patch
    center_box = (OFFSET, OFFSET, OFFSET + CENTER_SIZE, OFFSET + CENTER_SIZE)
    distorted_center = distorted_patch.crop(center_box)
    fused_patch = clean_patch.copy()
    fused_patch.paste(distorted_center, center_box)

    # --- Save patches ---
    clean_patch_filename = f"{img_id_base}_patch_{x}_{y}_clean.png"
    fused_patch_filename = f"{img_id_base}_patch_{x}_{y}_fused.png"
    clean_patch.save(os.path.join(patch_image_path, clean_patch_filename), "PNG")
    fused_patch.save(os.path.join(patch_image_path, fused_patch_filename), "PNG")

    # --- Assign patch score (severity in [0,1]) ---
    patch_score = float(grades[level_idx])

    # Metadata entry
    patch_entry = {
        "unique_sample_id": f"{img_id_base}_{x}_{y}",
        "clean_image": clean_patch_filename,
        "distorted_image": fused_patch_filename,
        "score": patch_score,
        "metadata": {
            "image_id": img_id,
            "compression_type": comp_type,
            "ratings": ratings,
            "region": patch_region,
            "method": "synthetic_sampling",
        },
    }
    all_patch_metadata.append(patch_entry)

    # --- Compute PSNR ---
    np_clean = np.array(clean_patch).astype(np.float64)
    np_fused = np.array(fused_patch).astype(np.float64)
    mse = np.mean((np_clean - np_fused) ** 2)
    psnr_value = 120.0 if mse == 0 else 10 * math.log10(255**2 / mse)
    all_psnr_values.append(psnr_value)
    patch_entry["metadata"]["psnr"] = psnr_value


# --- Save metadata JSON ---
output_json_path = os.path.join(patch_image_path, "metadata.json")
with open(output_json_path, "w") as f:
    json.dump(all_patch_metadata, f, indent=4)

# --- PSNR histogram ---
plt.figure(figsize=(10, 6))
plt.hist(all_psnr_values, bins=50, edgecolor="black")
plt.title("Histogram of PSNR Values for Synthetic Patches")
plt.xlabel("PSNR (dB)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(patch_image_path, "psnr_histogram.png"))
