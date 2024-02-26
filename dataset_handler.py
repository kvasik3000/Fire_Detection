from typing import Literal, Optional
from os.path import join
from pathlib import Path

import argparse
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

import fiftyone.zoo as foz
import fiftyone as fo


class YOLO_Dataset(Dataset):
    """
    Класс для работы с датасетом YOLO.
    * Обрабатывает тренировочную, валидационную и тестовую части, пути до которых берутся из data.yaml.
    * При необходимости может работать с отдельной частью датасета, либо только с указанной директорией.
    * Предоставляет возможность найти статистику по указанной части датасета и позволяет проверить нормализацию.
    """
    def __init__(self, 
                 dataset_dir: str, 
                 part: Optional[Literal['train', 'val', 'test', 'all']] = 'all',
                 no_yolo: bool = False, 
                 **transform_kwargs):
        """
        Parameters
        ---------- 
        dataset_dir: str
            Путь до датасета.

        part: ['train', 'val', 'test', 'all']
            Часть датасета, которая будет обрабатываться. 
            По умолчанию обрабатывается весь датасет. 

        no_yolo: bool
            Работать только с директорией, указанной в dataset_dir.

        transform_kwargs
            Параметры для преобразования изображений перед работой с ними.    
        """ 
        self.dataset_dir = dataset_dir
        self.part = part
        self.no_yolo = no_yolo

        self.resize_size = transform_kwargs.get('resize', 640)
        self.mean = transform_kwargs.get('mean')
        self.std = transform_kwargs.get('std')

        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)

        if not self.no_yolo:
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

        else:
            self.images = np.array([join(self.dataset_dir, img_name) for img_name in os.listdir(self.dataset_dir)])

        self.transform = v2.Compose([
            v2.Resize(self.resize_size),
            v2.ToTensor(),
            v2.Normalize(mean = self.mean, std = self.std),
        ])

    def change_part(self, part: Literal['train', 'val', 'test', 'all']):
        """
        Изменяет выбранную часть датасета для дальнейшей работы.

        Parameters
        ---------- 
        part: ['train', 'val', 'test', 'all']
            Часть датасета, с которой будет работать класс.  
        """ 
        if self.no_yolo:
            return

        if part == 'all':
            self.images = np.concatenate([self.images_paths[key] for key in self.images_paths.keys()], axis = 0)
        else:
            self.images = self.images_paths[part] 
    
    def find_statistic(self, transform = None, save: bool = True):
        """
        Находит среднее и стандартное отклонение по указанной части датасета.

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

        if save:          
            self.mean = mean
            self.std = std

            self.transform = self._update_transform(self.mean, self.std)

            print(f'mean = {mean}')
            print(f'std = {std}')

        return mean, std

    def check_normalization(self, mean: ArrayLike = None, std: ArrayLike = None):
        """
        Показывает среднее и стандартное отклонение после нормализации указанной части датасета.

        Parameters
        ---------- 
        mean: ArrayLike
            Среднее для нормализации. Если None, то используется сохраненное.

        std: ArrayLike
            Стандартное отклонение для нормализации. Если None, то используется сохраненное.
        """    
        cur_mean = self.mean if mean is None else mean
        cur_std  = self.std if std is None else std

        assert cur_mean is not None and cur_std is not None, 'Необходимо сначала рассчитать статистику!'

        print(f'Текущее mean = {cur_mean}')
        print(f'Текущее std = {cur_std}')

        mean, std = self.find_statistic(self._update_transform(cur_mean, cur_std), save = False)

        print(f'mean после нормализации = {mean}')
        print(f'std после нормализации = {std}')

    def _update_transform(self, mean: ArrayLike = None, std: ArrayLike = None):

        new_transform = self.transform

        for idx, transform in enumerate(new_transform.transforms):
            if isinstance(transform, v2.Normalize):
                new_transform.transforms[idx] = v2.Normalize(mean, std)
                break

        return new_transform

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

        for sample in tqdm(self.dataset, desc = 'Присвоение тегов'):
            for part in ['train', 'val', 'test']:
                if part in sample.filepath:
                    sample.tags.append(part)
                    sample.save()
                    break


        self.model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
     

    def find_duplicates(self, cosine_treshold: float = 0.99):
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

        for idx, sample in enumerate(tqdm(self.dataset, desc = 'Обозначение дубликатов')):
            if sample.filepath not in self.samples_to_remove:

                dup_idxs = np.where(similarity_matrix[idx] > cosine_treshold)[0]

                for dup in dup_idxs:
                    self.samples_to_remove.add(path_map[dup])

                if len(dup_idxs) > 0:
                    sample.tags.append("Имеет дубликат")
                    sample.save()

            else:
                sample.tags.append("Дубликат")
                sample.save()

    def show(self):
        session = fo.launch_app(self.dataset)
        session.show()

    def delete_duplicates(self):
        
        for path in self.samples_to_remove: 
            os.remove(Path(path.replace('images', 'labels', 1)).with_suffix('.txt'))
            os.remove(path)
        
        print(f'Удалено {len(list(self.samples_to_remove))} изображений')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help = True)

    parser.add_argument('--delete_duplicates', action = 'store_true', help = 'Удалить дубликаты')
    parser.add_argument('--find_statistic', action = 'store_true', help = 'Рассчитать статистику')
    parser.add_argument('--check_normalization', action = 'store_true', help = 'Проверить нормализацию с рассчитанной или заданной статистикой')
    parser.add_argument('-m', '--mean', type = float, nargs = "*", dest = 'mean', help = 'Среднее для нормализации', default = None)
    parser.add_argument('-s', '--std', type = float, nargs = "*", dest = 'std', help = 'Стандартное отклонение для нормализации', default = None)
    parser.add_argument('-p', '--path', type = str, dest = 'path', help = 'Путь целовой директории', default = None)
    parser.add_argument('--yolo_dir', action = 'store_true', help = 'Считать целевую директорию датасетом YOLO', default = False)
    parser.add_argument('--yolo_part', type = str, dest = 'part', help = 'Часть датасета YOLO для работы с ней: ["train", "val", "test", "all"]', default = "all")
    parser.add_argument('-t', '--treshold', type = float, dest = 'treshold', help = 'Порог косинусного расстояния для определения дубликатов', default = 0.99)

    args = parser.parse_args()

    dataset = YOLO_Dataset(dataset_dir = args.path,
                           part = args.part,
                           no_yolo = not args.yolo_dir)
    
    if args.find_statistic:
        dataset.find_statistic()

    if args.check_normalization:
        dataset.check_normalization(mean = args.mean, std = args.std)

    if args.delete_duplicates:
        handler = Duplicates_handler(dataset)
        handler.find_duplicates(args.treshold)
        handler.delete_duplicates()
    