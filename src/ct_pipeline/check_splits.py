import os

BASE_DIR = r"D:\LiverAI_Project"
SPLIT_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "splits")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")
    for cls in ["Normal", "Tumor"]:
        folder = os.path.join(SPLIT_DIR, split, cls)
        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        print(f"{cls}: {count}")