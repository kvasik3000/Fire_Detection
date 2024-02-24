from typing import Literal, Optional
from os.path import join

import yaml
import os

from numpy.typing import ArrayLike  
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as v2
import torch

from PIL import Image 

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from fiftyone import ViewField as F
import fiftyone.zoo as foz
import fiftyone as fo


class YOLO_Dataset(Dataset):
    """
    Класс для работы с датасетом YOLO.
    Обрабатывает тренировочную, валидационную и тестовую части, пути до которых берутся из data.yaml.
    При необходимости может работать с отдельной частью датасета.
    Автоматически находит статистику по указанной части датасета и позволяет проверить нормализацию.
    """
    def __init__(self, dataset_dir: str, part: Optional[Literal['train', 'val', 'test', 'all']] = 'all', **transform_kwargs):
        """
        Parameters
        ---------- 
        dataset_dir: str
            Путь до датасета.

        part: ['train', 'val', 'test', 'all']
            Часть датасета, которая будет обрабатываться. 
            По умолчанию обрабатывается весь датасет. 

        transform_kwargs
            Параметры для преобразования изображений перед работой с ними.    
        """ 

        self.dataset_dir = dataset_dir
        self.part = part

        self.resize_size = transform_kwargs.get('resize', 640)

        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)

        # Данные из data.yaml файла
        data = None
            
        # Считывание данных с yaml файла
        with open(join(self.dataset_dir, 'data.yaml'), 'r') as file:
            data = yaml.safe_load(file)

        # Пути к тренировочной, валидационной и тестовой частям в датасете из yaml файла
        paths = {dataset_part : join(dataset_dir, data[dataset_part].strip('./')) for dataset_part in ['train', 'val', 'test']}
        self.images_paths = {part : np.array([join(paths[part], img_name) for img_name in os.listdir(paths[part])]) for part in paths.keys()}

        # Указание текущих обрабатываемых изображений
        self.images = None
        self.change_part(self.part)

        self.transform = v2.Compose([
            v2.Resize(self.resize_size),
            v2.ToTensor(),
            v2.Normalize(mean = self.mean, std = self.std),
        ])

    def change_part(self, part: Literal['train', 'val', 'test', 'all']):
        """
        Parameters
        ---------- 
        part: ['train', 'val', 'test', 'all']
            Часть датасета, с которой будет работать класс.  
        """ 

        if part == 'all':
            self.images = np.concatenate([self.images_paths[key] for key in self.images_paths.keys()], axis = 0)
        else:
            self.images = self.images_paths[part] 

        # Нахождение статистики по указанной части датасета
        self.mean, self.std = self.find_statistic()
    
    def find_statistic(self, transform = None):
        """
        Находит среднее и стандартное отклонение по указанной части датасета.

        Parameters
        ---------- 
        transform
            Преобразование изображений перед поиском статистики.

        Returns
        ---------- 
        mean: ArrayLike
            Среднее по указанной части датасета.
        
        std: ArrayLike
            Стандартное отклонение по указанной части датасета.
        """
        mean = torch.zeros(3)
        std  = torch.zeros(3)

        cnt = len(self.images) * self.resize_size[0] ** 2 

        if transform is None:
            transform = v2.Compose([
                v2.Resize(self.resize_size),
                v2.ToTensor()])

        for image_path in tqdm(self.images, desc = 'Расчёт статистики'):
            
            image = transform(Image.open(image_path).convert('RGB'))

            mean += image.sum(axis = [1, 2])
            std  += (image ** 2).sum(axis = [1, 2])
 
        mean /= cnt
        std  = torch.sqrt(std / cnt - mean ** 2) 

        return mean, std
    
    def check_normalization(self, mean: ArrayLike = None, std: ArrayLike = None):
        """
        Показывает среднее и стандартное отклонение после нормализации указанной части датасета.

        Parameters
        ---------- 
        mean: ArrayLike
            Среднее для нормализации. Если None, то используется автоматически
            рассчитанное.

        std: ArrayLike
            Стандартное отклонение для нормализации. Если None, то используется автоматически
            рассчитанное.
        """    

        mean = self.mean if mean is None else mean
        std  = self.std if std is None else std

        print(f'Текущее mean = {mean}')
        print(f'Текущее std = {std}')

        mean, std = self.find_statistic(self.transform)

        print(f'mean после нормализации = {mean}')
        print(f'std после нормализации = {std}')

    def __getitem__(self, id):

        image = Image.open(self.images[id]).convert('RGB')
        image = self.transform(image)

        return image
    
    def __len__(self):
        return len(self.images)
    
    
class Duplicates_handler:
    """
    Обработчик дубликатов в YOLO датасете. Проверяет на дубликаты директории с тренировочным,
    валидационным и тестовым датасетами, а также позволяет их визуализировать и устранить.
    """
    def __init__(self, dataset: YOLO_Dataset):
        """
        Parameters
        ---------- 
        dataset: YOLO_Dataset
            Экземпляр класса YOLO_Dataset.
        """ 

        self.dataset = fo.Dataset.from_images(dataset.images)

        for sample in self.dataset:
            for part in ['train', 'val', 'test']:
                if part in sample.filepath:

                    sample.tags.append(part)
                    sample.save()

                    break


        self.model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
     

    def find_duplicates(self, cosine_treshlod: float = 0.99):
        """
        Определяет дубликаты в датасете путем создания эмбеддингов и определения косинусного расстояния между ними.

        Parameters
        ---------- 
        cosine_treshlod: float
            Порог косинусного расстояния для определения дубликатов.
        """

        # Нахождение эмбеддингов с помощью модели
        embeddings = self.dataset.compute_embeddings(self.model)

        # Вычисление косинусного сходства для эмбеддингов
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = similarity_matrix - np.identity(len(similarity_matrix))

        # Обозначение дубликатов
        path_map = [s.filepath for s in self.dataset.select_fields(["filepath"])]
                    
        self.samples_to_remove = set()

        for idx, sample in enumerate(self.dataset):
            if sample.filepath not in self.samples_to_remove:

                dup_idxs = np.where(similarity_matrix[idx] > cosine_treshlod)[0]

                for dup in dup_idxs:
                    self.samples_to_remove.add(path_map[dup])

                if len(dup_idxs) > 0:
                    sample.tags.append("Имеет дубликат")
                    sample.save()

            else:
                sample.tags.append("Дубликат")
                sample.save()

    def show_duplicates(self):
        session = fo.launch_app(self.dataset)
        session.show()

    def delete_duplicates(self):
        
        for path in self.samples_to_remove:
            os.remove(path)
        
        print(f'Удалено {len(list(self.samples_to_remove))} изображений')