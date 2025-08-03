import pandas as pd
import numpy as np
from PIL import Image
import json
import os
import random
import math
import matplotlib.pyplot as plt

source_image_path = "source_image"
jpeg_image_path = "jpeg"
bpg_image_path = "bpg"
patch_image_path = "patches"

metadata_file = pd.read_csv("subjective_ratings.csv")

all_patch_metadata = []
all_psnr_values = []

for index, row_data in metadata_file.iterrows():
    img_id = row_data["image_id"]
    img_id_base = os.path.splitext(img_id)[0]
    comp_type = row_data["Compression type"]
    number_of_ratings = row_data["No. of ratings"]
    mean = row_data["mean"]
    std = row_data["std"]
    ratings = json.loads(row_data["ratings"])

    sample_id = f"id_{img_id_base}_{comp_type}"
    clean_file = f"{sample_id}_clean.png"
    dist_file = f"{sample_id}_fused.png"

    source_image_file_name = f"{img_id}"
    source_image_full_path = os.path.join(source_image_path, source_image_file_name)
    if comp_type == "JPEG":
        distorted_filename = f"{img_id_base}_JPEG_0{int(mean):02d}.jpg"
        distorted_image_path = os.path.join(jpeg_image_path, distorted_filename)
        normalized_score = (mean - 1) / 99
    else:
        distorted_filename = f"{img_id_base}_BPG_0{int(mean):02d}.png"
        distorted_image_path = os.path.join(bpg_image_path, distorted_filename)
        normalized_score = (mean - 1) / 50

    source_image = Image.open(source_image_full_path)
    distorted_image = Image.open(distorted_image_path)
    PATCH_SIZE = 20
    k = 1
    max_x = source_image.width - PATCH_SIZE
    max_y = source_image.height - PATCH_SIZE

    for i in range(k):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        patch_region = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
        source_patch = source_image.crop(patch_region)
        distorted_patch = distorted_image.crop(patch_region)

        CENTER_SIZE = 12
        OFFSET = (PATCH_SIZE - CENTER_SIZE) // 2
        center_box = (OFFSET, OFFSET, OFFSET + CENTER_SIZE, OFFSET + CENTER_SIZE)
        patch_center = source_patch.crop(center_box)
        fused_distorted_patch = distorted_patch.copy()
        fused_distorted_patch.paste(patch_center, center_box)

        clean_patch_filename = f"{sample_id}_patch_{x}_{y}_clean.png"
        fused_distorted_patch_filename = f"{sample_id}_patch_{x}_{y}_fused.png"
        source_patch.save(os.path.join(patch_image_path, clean_patch_filename), "PNG")
        fused_distorted_patch.save(
            os.path.join(patch_image_path, fused_distorted_patch_filename), "PNG"
        )

        np_clean_patch = np.array(source_patch).astype(np.float64)
        np_fused_patch = np.array(fused_distorted_patch).astype(np.float64)
        MAX_PIXEL_VALUE = 255.0

        mse = np.mean((np_clean_patch - np_fused_patch) ** 2)

        psnr_value = float("inf")
        if mse != 0:
            psnr_value = 10 * math.log10((MAX_PIXEL_VALUE**2) / mse)
        all_psnr_values.append(psnr_value)

        sample_entry = {
            "unique_sample_id": sample_id,
            "clean_image": clean_patch_filename,
            "distorted_image": fused_distorted_patch_filename,
            "score": normalized_score,
            "metadata": {
                "image_id": img_id,
                "Compression type": comp_type,
                "No. of ratings": number_of_ratings,
                "mean": mean,
                "std": std,
                "ratings": ratings,
                "region": patch_region,
                "psnr": psnr_value,
                "method": "new_ref",
                "src_image": img_id_base,
            },
        }
        all_patch_metadata.append(sample_entry)

output_json_path = os.path.join(patch_image_path, "metadata.json")
with open(output_json_path, "w") as f:
    json.dump(all_patch_metadata, f, indent=4)

finite_psnr_values = [p for p in all_psnr_values if p != float("inf")]
print(f"Total PSNR values collected: {len(all_psnr_values)}")
print(f"Finite PSNR values: {len(finite_psnr_values)}")
if finite_psnr_values:
    print(f"Example PSNR values (first 5): {finite_psnr_values[:5]}")
histogram_save_path = os.path.join(patch_image_path, "psnr_histogram.png")

plt.figure(figsize=(10, 6))
plt.hist(finite_psnr_values, bins=50, edgecolor="black")
plt.title("Histogram of PSNR Values for Fused Patches")
plt.xlabel("PSNR (dB)")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.savefig(histogram_save_path)
