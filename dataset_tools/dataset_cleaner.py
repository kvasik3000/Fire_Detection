import os

from tqdm.auto import tqdm

DATASET_FOLDER = "/ssd/babenko/fire-detection/datasets/fire_1"

FOLDERS = ["test", "train", "valid"]
TARGET_FOLDERS = ["labels"]

# CLASSES_RENAME = {
#     "0": None,
#     "1": "0",
# }

CLASSES_RENAME = {"1": "0"}

for folder in FOLDERS:
    for target in TARGET_FOLDERS:
        folder_path = os.path.join(DATASET_FOLDER, folder, target)
        files = sorted(os.listdir(folder_path), reverse=True)

        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)
            file_data = ""
            with open(file_path) as f:
                while True:
                    line = f.readline().strip()
                    if line == "":
                        break
                    class_name = line.split(" ")[0]
                    bbox_data = " ".join(line.split(" ")[1:])
                    newname = CLASSES_RENAME.get(class_name, None)
                    if newname:
                        file_data += newname + " " + bbox_data + "\n"
            with open(file_path, "w") as f:
                f.write(file_data)
