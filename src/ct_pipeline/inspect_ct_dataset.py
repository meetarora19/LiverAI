import os
import json
import nibabel as nib
import numpy as np

BASE_DIR = r"D:\LiverAI_Project"

DATASET_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "raw", "Task03_Liver")
IMAGES_TR = os.path.join(DATASET_DIR, "imagesTr")
LABELS_TR = os.path.join(DATASET_DIR, "labelsTr")
IMAGES_TS = os.path.join(DATASET_DIR, "imagesTs")
DATASET_JSON = os.path.join(DATASET_DIR, "dataset.json")


def list_clean_nii_files(folder_path):
    return sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ])


def load_metadata():
    if not os.path.exists(DATASET_JSON):
        return None
    with open(DATASET_JSON, "r") as f:
        return json.load(f)


def count_slices_with_labels(label_volume):
    liver_slices = 0
    tumor_slices = 0

    depth = label_volume.shape[2]
    for i in range(depth):
        slice_lbl = label_volume[:, :, i]
        if np.any(slice_lbl == 1):
            liver_slices += 1
        if np.any(slice_lbl == 2):
            tumor_slices += 1

    return liver_slices, tumor_slices


def compute_case_summary(image_path, label_path):
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image_data = image_nii.get_fdata()
    label_data = label_nii.get_fdata()

    spacing = image_nii.header.get_zooms()  # (x, y, z)
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    liver_voxels = int(np.sum(label_data == 1))
    tumor_voxels = int(np.sum(label_data == 2))

    liver_volume_cm3 = (liver_voxels * voxel_volume_mm3) / 1000.0
    tumor_volume_cm3 = (tumor_voxels * voxel_volume_mm3) / 1000.0

    liver_slices, tumor_slices = count_slices_with_labels(label_data)

    has_tumor = tumor_voxels > 0

    summary = {
        "shape": image_data.shape,
        "spacing": spacing,
        "num_slices": image_data.shape[2],
        "liver_slices": liver_slices,
        "tumor_slices": tumor_slices,
        "has_tumor": has_tumor,
        "liver_voxels": liver_voxels,
        "tumor_voxels": tumor_voxels,
        "liver_volume_cm3": liver_volume_cm3,
        "tumor_volume_cm3": tumor_volume_cm3
    }

    return summary


def print_dataset_overview():
    train_images = list_clean_nii_files(IMAGES_TR)
    train_labels = list_clean_nii_files(LABELS_TR)
    test_images = list_clean_nii_files(IMAGES_TS)

    print("\n========== CT DATASET OVERVIEW ==========\n")
    print(f"Training CT scans : {len(train_images)}")
    print(f"Training masks    : {len(train_labels)}")
    print(f"Test CT scans     : {len(test_images)}")

    meta = load_metadata()
    if meta:
        print("\nDataset Info:")
        print(f"Name             : {meta.get('name')}")
        print(f"Description      : {meta.get('description')}")
        print(f"Image Type       : {meta.get('tensorImageSize')} CT volumes")
        print(f"Training Cases   : {meta.get('numTraining')}")
        print(f"Test Cases       : {meta.get('numTest')}")

        labels = meta.get("labels", {})
        if labels:
            print("\nLabel Meaning:")
            for k, v in labels.items():
                print(f"  {k} -> {v}")


def inspect_first_case():
    train_images = list_clean_nii_files(IMAGES_TR)
    if not train_images:
        print("No training scans found.")
        return

    first_case = train_images[0]
    image_path = os.path.join(IMAGES_TR, first_case)
    label_path = os.path.join(LABELS_TR, first_case)

    if not os.path.exists(label_path):
        print(f"Matching mask not found for {first_case}")
        return

    summary = compute_case_summary(image_path, label_path)

    print("\n========== SAMPLE TRAINING CASE ==========\n")
    print(f"Case ID                  : {first_case}")
    print(f"Scan Size                : {summary['shape'][0]} x {summary['shape'][1]} x {summary['shape'][2]}")
    print(f"Number of Slices         : {summary['num_slices']}")
    print(f"Voxel Spacing (mm)       : {summary['spacing']}")
    print(f"Slices containing liver  : {summary['liver_slices']}")
    print(f"Slices containing tumor  : {summary['tumor_slices']}")
    print(f"Tumor present?           : {'Yes' if summary['has_tumor'] else 'No'}")
    print(f"Liver Volume (approx)    : {summary['liver_volume_cm3']:.2f} cm³")
    print(f"Tumor Volume (approx)    : {summary['tumor_volume_cm3']:.2f} cm³")


def inspect_dataset_statistics(max_cases=10):
    train_images = list_clean_nii_files(IMAGES_TR)

    if not train_images:
        print("No training scans found.")
        return

    print("\n========== QUICK DATASET STATISTICS ==========\n")
    print(f"Inspecting first {min(max_cases, len(train_images))} training cases...\n")

    tumor_case_count = 0
    total_liver_slices = 0
    total_tumor_slices = 0

    liver_volumes = []
    tumor_volumes = []

    for case_file in train_images[:max_cases]:
        image_path = os.path.join(IMAGES_TR, case_file)
        label_path = os.path.join(LABELS_TR, case_file)

        if not os.path.exists(label_path):
            continue

        summary = compute_case_summary(image_path, label_path)

        if summary["has_tumor"]:
            tumor_case_count += 1

        total_liver_slices += summary["liver_slices"]
        total_tumor_slices += summary["tumor_slices"]

        liver_volumes.append(summary["liver_volume_cm3"])
        tumor_volumes.append(summary["tumor_volume_cm3"])

    inspected = min(max_cases, len(train_images))

    print(f"Cases inspected                  : {inspected}")
    print(f"Cases with tumor                 : {tumor_case_count}")
    print(f"Average liver slices per case    : {total_liver_slices / inspected:.2f}")
    print(f"Average tumor slices per case    : {total_tumor_slices / inspected:.2f}")
    print(f"Average liver volume             : {np.mean(liver_volumes):.2f} cm³")
    print(f"Average tumor volume             : {np.mean(tumor_volumes):.2f} cm³")


def inspect_first_test_case():
    test_images = list_clean_nii_files(IMAGES_TS)
    if not test_images:
        print("No test scans found.")
        return

    test_case = test_images[0]
    test_path = os.path.join(IMAGES_TS, test_case)

    test_nii = nib.load(test_path)
    test_data = test_nii.get_fdata()
    spacing = test_nii.header.get_zooms()

    print("\n========== SAMPLE TEST CASE ==========\n")
    print(f"Case ID                  : {test_case}")
    print(f"Scan Size                : {test_data.shape[0]} x {test_data.shape[1]} x {test_data.shape[2]}")
    print(f"Number of Slices         : {test_data.shape[2]}")
    print(f"Voxel Spacing (mm)       : {spacing}")
    print("Tumor label not available for test set.")


def main():
    print_dataset_overview()
    inspect_first_case()
    inspect_dataset_statistics(max_cases=10)
    inspect_first_test_case()
    print("\n======================================\n")


if __name__ == "__main__":
    main()