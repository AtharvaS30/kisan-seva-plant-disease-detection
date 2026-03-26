import os
import random
import shutil
from pathlib import Path

# ==============================
# CONFIG
# ==============================
SOURCE_DIR = "PlantVillage"      # original dataset
OUTPUT_DIR = "PlantVillage_split"  # new split dataset

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # for reproducibility

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================
for split in ["train", "val", "test"]:
    Path(os.path.join(OUTPUT_DIR, split)).mkdir(parents=True, exist_ok=True)

# ==============================
# SPLIT EACH CLASS
# ==============================
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path) or class_name == "PlantVillage":
        continue

    images = []
    for f in os.listdir(class_path):
        try:
            if os.path.isfile(os.path.join(class_path, f)):
                images.append(f)
        except PermissionError:
            print(f"Skipping {os.path.join(class_path, f)} due to PermissionError")
            continue
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    # Copy images to new folders
    for split_name, split_images in splits.items():
        split_class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        Path(split_class_dir).mkdir(parents=True, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy2(src, dst)

print("✅ Dataset successfully split into train / val / test!")
