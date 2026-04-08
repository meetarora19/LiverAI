import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

LIVER_MASK_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "masks", "liver")
TUMOR_MASK_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "masks", "tumor")


def get_bounding_box_dimensions(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0, 0
    x, y, w, h = cv2.boundingRect(coords)
    return w, h


def analyze_slice(slice_name: str):
    liver_mask_path = os.path.join(LIVER_MASK_DIR, slice_name)
    tumor_mask_path = os.path.join(TUMOR_MASK_DIR, slice_name)

    if not os.path.exists(liver_mask_path):
        raise FileNotFoundError(f"Liver mask not found: {liver_mask_path}")

    if not os.path.exists(tumor_mask_path):
        raise FileNotFoundError(f"Tumor mask not found: {tumor_mask_path}")

    liver_mask = cv2.imread(liver_mask_path, cv2.IMREAD_GRAYSCALE)
    tumor_mask = cv2.imread(tumor_mask_path, cv2.IMREAD_GRAYSCALE)

    if liver_mask is None:
        raise ValueError(f"Could not read liver mask: {liver_mask_path}")
    if tumor_mask is None:
        raise ValueError(f"Could not read tumor mask: {tumor_mask_path}")

    liver_mask = (liver_mask > 0).astype(np.uint8)
    tumor_mask = (tumor_mask > 0).astype(np.uint8)

    liver_area = int(liver_mask.sum())
    tumor_area = int(tumor_mask.sum())

    tumor_to_liver_ratio = (tumor_area / liver_area) if liver_area > 0 else 0.0

    tumor_w, tumor_h = get_bounding_box_dimensions((tumor_mask * 255).astype(np.uint8))

    result = {
        "slice_name": slice_name,
        "liver_area_pixels": liver_area,
        "tumor_area_pixels": tumor_area,
        "tumor_to_liver_ratio": round(tumor_to_liver_ratio, 6),
        "tumor_width_pixels": tumor_w,
        "tumor_height_pixels": tumor_h
    }

    print("\nMorphology Analysis")
    print("-------------------")
    print(f"Slice: {slice_name}")
    print(f"Liver Area (pixels): {liver_area}")
    print(f"Tumor Area (pixels): {tumor_area}")
    print(f"Tumor-to-Liver Ratio: {tumor_to_liver_ratio:.6f}")
    print(f"Tumor Width (pixels): {tumor_w}")
    print(f"Tumor Height (pixels): {tumor_h}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ct_pipeline/morphology_analysis.py <slice_png_name>")
    else:
        analyze_slice(sys.argv[1])