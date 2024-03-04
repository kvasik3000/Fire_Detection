import os
from hashlib import md5
from mmap import ACCESS_READ
from mmap import mmap
from pathlib import Path

DATASET_PATH = Path("/ssd/babenko/fire-detection/datasets/actual")

TARGET_FOLDERS = ["train", "valid", "test"]
FOLDER_INSIDE = ["images"]
SUB_FOLDER = ["labels"]

hash_table = {}

for folder in TARGET_FOLDERS:
    saved = 0
    fromother = 0
    deleted = 0
    for target, sub in zip(FOLDER_INSIDE, SUB_FOLDER):
        target_path = DATASET_PATH / folder / target
        sub_path = DATASET_PATH / folder / sub
        files = os.listdir(target_path)

        for file in files:
            image_path = target_path / file
            labelname, _ = os.path.splitext(file)
            label_path = sub_path / (labelname + ".txt")

            with open(image_path) as file, mmap(file.fileno(), 0, access=ACCESS_READ) as file:
                hashmd5 = md5(file).hexdigest()

            val = hash_table.get(hashmd5, None)
            if val:
                if val[-1] != folder:
                    fromother += 1
                os.remove(image_path)
                os.remove(label_path)
                deleted += 1
            else:
                hash_table.update({hashmd5: (image_path, label_path, folder)})
                saved += 1

    print(f"[{folder}] Proccesed: {saved+deleted} | Saved: {saved} | Deleted: {deleted}/ FromOther: {fromother}")
