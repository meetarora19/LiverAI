import os
import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

BASE_DIR = r"D:\LiverAI_Project"

RAW_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "raw", "Task03_Liver")
IMAGES_DIR = os.path.join(RAW_DIR, "imagesTr")
LABELS_DIR = os.path.join(RAW_DIR, "labelsTr")

SLICES_NORMAL_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "slices", "Normal")
SLICES_TUMOR_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "slices", "Tumor")

MASK_LIVER_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "masks", "liver")
MASK_TUMOR_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "masks", "tumor")

IMG_SIZE = 224

os.makedirs(SLICES_NORMAL_DIR, exist_ok=True)
os.makedirs(SLICES_TUMOR_DIR, exist_ok=True)
os.makedirs(MASK_LIVER_DIR, exist_ok=True)
os.makedirs(MASK_TUMOR_DIR, exist_ok=True)


def normalize_ct_slice(slice_2d):
    slice_2d = np.clip(slice_2d, -200, 250)
    slice_min = slice_2d.min()
    slice_max = slice_2d.max()

    if slice_max - slice_min < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    slice_2d = (slice_2d - slice_min) / (slice_max - slice_min)
    slice_2d = (slice_2d * 255).astype(np.uint8)
    return slice_2d


def save_png(path, arr):
    cv2.imwrite(path, arr)


def process_case(image_file):
    image_path = os.path.join(IMAGES_DIR, image_file)
    label_path = os.path.join(LABELS_DIR, image_file)

    if not os.path.exists(label_path):
        print(f"Skipping {image_file} — label not found")
        return 0, 0

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image_data = image_nii.get_fdata()
    label_data = label_nii.get_fdata()

    normal_count = 0
    tumor_count = 0

    depth = image_data.shape[2]

    for i in range(depth):
        img_slice = image_data[:, :, i]
        lbl_slice = label_data[:, :, i]

        liver_mask = (lbl_slice == 1).astype(np.uint8)
        tumor_mask = (lbl_slice == 2).astype(np.uint8)

        liver_pixels = int(liver_mask.sum())
        tumor_pixels = int(tumor_mask.sum())

        if liver_pixels == 0 and tumor_pixels == 0:
            continue

        img_slice = normalize_ct_slice(img_slice)
        img_slice = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        liver_mask_resized = cv2.resize(
            liver_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
        ) * 255

        tumor_mask_resized = cv2.resize(
            tumor_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
        ) * 255

        case_id = image_file.replace(".nii.gz", "")
        base_name = f"{case_id}_slice_{i:03d}.png"

        save_png(os.path.join(MASK_LIVER_DIR, base_name), liver_mask_resized)
        save_png(os.path.join(MASK_TUMOR_DIR, base_name), tumor_mask_resized)

        if tumor_pixels > 0:
            save_png(os.path.join(SLICES_TUMOR_DIR, base_name), img_slice)
            tumor_count += 1
        else:
            save_png(os.path.join(SLICES_NORMAL_DIR, base_name), img_slice)
            normal_count += 1

    return normal_count, tumor_count


def main():
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ])

    total_normal = 0
    total_tumor = 0

    print(f"Found {len(image_files)} training volumes")

    for image_file in tqdm(image_files, desc="Processing volumes"):
        try:
            n_count, t_count = process_case(image_file)
            total_normal += n_count
            total_tumor += t_count
        except Exception as e:
            print(f"\nSkipping {image_file} due to error: {e}")

    print("\nProcessing complete.")
    print(f"Saved Normal slices: {total_normal}")
    print(f"Saved Tumor slices: {total_tumor}")
    print(f"Normal folder: {SLICES_NORMAL_DIR}")
    print(f"Tumor folder: {SLICES_TUMOR_DIR}")


if __name__ == "__main__":
    main()