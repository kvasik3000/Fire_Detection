import os
import random

DATASET_FOLDER = "C:/Users/USER/Documents/nsu/4/smoke_trainset/5"
FOLDERS = ["test", "train", "valid"]

# удаление 80% от числа изображений, на которых нет целевого объекта

for folder in FOLDERS:
    count = 0

    labels_dir = os.path.join(DATASET_FOLDER, folder, 'labels')
    images_dir = os.path.join(DATASET_FOLDER, folder, 'images')
    label_files = os.listdir(labels_dir)
    random.shuffle(label_files)

    empty_files = [f for f in label_files if os.path.getsize(os.path.join(labels_dir,f))==0]

    total_empty_files = len(empty_files)
    num_files_to_delete = int(0.8 * total_empty_files)

    print(folder)
    print("Всего пустых: ", total_empty_files)
    print("Удалим 80%: ", num_files_to_delete)
    print()

    for file in empty_files:
        if count == num_files_to_delete:
            break

        label_path = os.path.join(labels_dir, file)
        image_path = os.path.join(images_dir, file.replace('.txt', '.jpg'))

        if os.path.getsize(label_path) == 0:
            os.remove(label_path)
            os.remove(image_path)
            count += 1
