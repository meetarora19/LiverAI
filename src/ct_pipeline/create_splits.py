import os
import shutil
from sklearn.model_selection import train_test_split

BASE_DIR = r"D:\LiverAI_Project"

SOURCE_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "slices")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

CLASSES = ["Normal", "Tumor"]


def make_split_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)


def get_image_files(class_name):
    class_dir = os.path.join(SOURCE_DIR, class_name)
    files = [
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    return sorted(files)


def copy_files(file_list, split_name, class_name):
    target_dir = os.path.join(SPLIT_DIR, split_name, class_name)
    for file_path in file_list:
        fname = os.path.basename(file_path)
        shutil.copy2(file_path, os.path.join(target_dir, fname))


def main():
    make_split_dirs()

    for class_name in CLASSES:
        files = get_image_files(class_name)

        train_files, temp_files = train_test_split(
            files,
            test_size=(1 - TRAIN_RATIO),
            random_state=SEED,
            shuffle=True
        )

        val_files, test_files = train_test_split(
            temp_files,
            test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
            random_state=SEED,
            shuffle=True
        )

        copy_files(train_files, "train", class_name)
        copy_files(val_files, "val", class_name)
        copy_files(test_files, "test", class_name)

        print(f"\nClass: {class_name}")
        print(f"Train: {len(train_files)}")
        print(f"Val:   {len(val_files)}")
        print(f"Test:  {len(test_files)}")

    print("\nCT slice split completed successfully.")
    print("Split directory:", SPLIT_DIR)


if __name__ == "__main__":
    main()