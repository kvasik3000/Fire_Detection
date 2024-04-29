import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Literal
from typing import Optional
from typing import Union

import albumentations as A
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
from numpy.typing import ArrayLike
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.transforms import transforms
from torchvision.transforms import v2
from tqdm import tqdm


class Dataset_Handler:
    """Обработчик датасета.

    * Может работать с YOLO-датасетом, либо с конкретной папкой с изображениями.
    * Если используется YOLO-датасет, то пути до отдельных частей датасета берутся из data.yaml.
    * При необходимости может работать с отдельной частью YOLO-датасета.
    * Предоставляет возможность найти статистику по датасету и проверить нормализацию.
    * Предоставляет возможность найти дубликаты с заданным порогом и удалить их.
    * Предоставляет возможность применить аугментации к датасету и визуализировать их с помощью приложения FiftyOne.
    """

    def __init__(
        self,
        dataset_dir: str,
        yolo_dataset: bool = False,
        split: Optional[Literal["train", "val", "test", "all"]] = "all",
        **transform_kwargs,
    ):
        """
        Parameters
        ----------
        dataset_dir: str
            Путь до датасета.

        yolo_dataset: bool
            Является ли датасет YOLO-датасетом.

        split: ['train', 'val', 'test', 'all']
            Часть датасета, которая будет обрабатываться.
            По умолчанию обрабатывается весь датасет.

        transform_kwargs
            Параметры для преобразования изображений перед работой с ними.
        """
        self.split = split
        self.yolo_dataset = yolo_dataset

        self.dataset = fo.Dataset()
        self.images = None

        if self.yolo_dataset:
            for split in ["train", "val", "test"]:
                self.dataset.add_dir(
                    dataset_dir=dataset_dir, dataset_type=fo.types.YOLOv5Dataset, yaml_path="data.yaml", split=split, tags=split
                )

            self.change_split(self.split)

        else:
            self.dataset = self.dataset.from_images_dir(dataset_dir)
            self.change_split("all")

        self.resize_size = transform_kwargs.get("resize", 640)
        self.mean = transform_kwargs.get("mean")
        self.std = transform_kwargs.get("std")

        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)

    def change_split(self, split: Literal["train", "val", "test", "all"]):
        """Изменяет выбранную часть YOLO-датасета для дальнейшей работы.

        Parameters
        ----------
        part: ['train', 'val', 'test', 'all']
            Часть датасета, с которой будет работать класс.
        """
        if split == "all":
            self.images = self.dataset.values("filepath")
        else:
            self.images = self.dataset.select_by("tags", [split]).values("filepath")

    def find_statistic(self, transform: Union[v2.Compose, transforms.Compose, A.Compose] = None, save: bool = True):
        """Находит среднее и стандартное отклонение по датасету.

        Parameters
        ----------
        transform
            Преобразование изображений перед поиском статистики.

        save: bool
            Нужно ли сохранить в классе рассчитанную статистику.

        Returns
        ----------
        mean: ArrayLike
            Среднее по указанной части датасета.

        std: ArrayLike
            Стандартное отклонение по указанной части датасета.
        """
        mean = torch.zeros(3)
        std = torch.zeros(3)

        cnt = len(self.images) * self.resize_size[0] ** 2

        if transform is None:
            transform = v2.Compose([v2.Resize(self.resize_size), v2.ToTensor()])

        for image_path in tqdm(self.images, desc="Расчёт статистики"):
            image = Image.open(image_path).convert("RGB")

            if isinstance(transform, A.Compose):
                image = transform(image=np.array(image))["image"]
                image = Image.fromarray(image)
            else:
                image = transform(image)

            mean += image.sum(axis=[1, 2])
            std += (image**2).sum(axis=[1, 2])

        mean /= cnt
        std = torch.sqrt(std / cnt - mean**2)

        if save:
            self.mean = mean
            self.std = std

            print(f"mean = {mean}")
            print(f"std = {std}")

        return mean, std

    def check_normalization(self, mean: ArrayLike = None, std: ArrayLike = None):
        """Показывает среднее и стандартное отклонение после нормализации датасета.

        Parameters
        ----------
        mean: ArrayLike
            Среднее для нормализации. Если None, то используется сохраненное.

        std: ArrayLike
            Стандартное отклонение для нормализации. Если None, то используется сохраненное.
        """
        cur_mean = self.mean if mean is None else mean
        cur_std = self.std if std is None else std

        assert cur_mean is not None and cur_std is not None, "Необходимо сначала рассчитать статистику!"

        print(f"Текущее mean = {cur_mean}")
        print(f"Текущее std = {cur_std}")

        transform = v2.Compose([v2.Resize(self.resize_size), v2.ToTensor(), v2.Normalize(cur_mean, cur_std)])
        mean, std = self.find_statistic(transform, save=False)

        print(f"mean после нормализации = {mean}")
        print(f"std после нормализации = {std}")

    def find_duplicates(self, cosine_treshold: float = 0.99):
        """Определяет дубликаты в датасете путем создания эмбеддингов и определения
        косинусного расстояния между ними.

        Parameters
        ----------
        cosine_treshlod: float
            Порог косинусного расстояния для определения дубликатов.
        """
        model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
        self.dataset.add_sample_field("duplicate_id", fo.IntField)

        # Нахождение эмбеддингов с помощью модели
        embeddings = self.dataset.compute_embeddings(model)

        # Вычисление косинусного сходства для эмбеддингов
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = similarity_matrix - np.identity(len(similarity_matrix))

        # Обозначение дубликатов
        id_map = [s.id for s in self.dataset.select_fields(["id"])]

        self.duplicates = []
        cur_dup_id = 0

        for idx, sample in enumerate(self.dataset.iter_samples(progress=True)):
            if sample.filepath not in self.duplicates:

                dup_idxs = np.where(similarity_matrix[idx] > cosine_treshold)[0]

                if len(dup_idxs) > 0:

                    added = False

                    for dup in dup_idxs:
                        dup_sample = self.dataset[id_map[dup]]

                        if dup_sample.filepath not in self.duplicates:
                            dup_sample.tags.append("Дубликат")
                            dup_sample.duplicate_id = cur_dup_id
                            dup_sample.save()

                            self.duplicates.append(dup_sample.filepath)
                            added = True

                    if added:
                        sample.tags.append("Имеет дубликат")
                        sample.duplicate_id = cur_dup_id
                        sample.save()

                        cur_dup_id += 1

    def delete_duplicates(self, verbose: bool = False):
        """Удаляет найденные дубликаты из датасета.

        Parameters
        ----------
        verbose: bool
            Выводить информацию об удаленных файлах.
        """
        for path in self.duplicates:

            label = Path(path.replace("images", "labels", 1)).with_suffix(".txt")

            if self.yolo_dataset:
                os.remove(label)

                if verbose:
                    print(f"Удалена метка: {label}")

            os.remove(path)

            if verbose:
                print(f"Удалено изображение: {path}")

        print(f"Количество удаленных изображений: {len(self.duplicates)}")

    def show_duplicates(self):
        """Визуализация найденных дубликатов."""
        view = self.dataset.sort_by("duplicate_id", reverse=True)
        fo.launch_app(self.dataset, view=view)

    def show_augmentations(
        self, transform: Union[v2.Compose, transforms.Compose, A.Compose], existing_dir: str = None, seed: int = None
    ):
        """Визуализация датасета с примененными аугментациями.

        Parameters
        ----------
        transform: Compose
            Аугментации для применения к датасету.

        existing_dir: str
            Визуализирует изображения из указанной директории (без применения аугментаций).

        seed: int
            Сид для возпроизведения аугментаций.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if existing_dir is None:

            tmp_path = tempfile.mkdtemp(dir=".")

            for path in tqdm(self.images):
                image = Image.open(path).convert("RGB")

                if isinstance(transform, A.Compose):
                    image = transform(image=np.array(image))["image"]
                    image = Image.fromarray(image)
                else:
                    image = transform(image)

                sample_name = os.path.basename(path)
                image.save(os.path.join(tmp_path, sample_name))

            augmented_dataset = fo.Dataset.from_images_dir(tmp_path)

        else:
            augmented_dataset = fo.Dataset.from_images_dir(existing_dir)

        fo.launch_app(augmented_dataset)

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--delete_duplicates", action="store_true", help="Удалить дубликаты")

    parser.add_argument("--find_statistic", action="store_true", help="Рассчитать статистику")

    parser.add_argument(
        "--check_normalization", action="store_true", help="Проверить нормализацию с рассчитанной или заданной статистикой"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Выводить информацию об удаленных дубликатах")

    parser.add_argument("-m", "--mean", type=float, nargs="*", dest="mean", help="Среднее для нормализации", default=None)

    parser.add_argument(
        "-s", "--std", type=float, nargs="*", dest="std", help="Стандартное отклонение для нормализации", default=None
    )

    parser.add_argument("-p", "--path", type=str, dest="path", help="Путь целовой директории", default=None)

    parser.add_argument("--yolo_dir", action="store_true", help="Считать целевую директорию датасетом YOLO", default=False)

    parser.add_argument(
        "--yolo_split",
        type=str,
        dest="split",
        help='Часть датасета YOLO для работы с ней: ["train", "val", "test", "all"]',
        default="all",
    )

    parser.add_argument(
        "-t",
        "--treshold",
        type=float,
        dest="treshold",
        help="Порог косинусного расстояния для определения дубликатов",
        default=0.99,
    )

    args = parser.parse_args()

    dataset = Dataset_Handler(dataset_dir=args.path, split=args.split, yolo_dataset=args.yolo_dir)

    if args.find_statistic:
        dataset.find_statistic()

    if args.check_normalization:
        dataset.check_normalization(mean=args.mean, std=args.std)

    if args.delete_duplicates:
        dataset.find_duplicates(args.treshold)
        dataset.delete_duplicates(args.verbose)
