 <h1 align="center"> Детектор огня и дыма </h1>

---

<img src = "https://github.com/kvasik3000/Fire_Detection/blob/main/images/fire.jpg" width = 1000px height = 360px align='center'>

---

<img src = "https://img.shields.io/badge/Python 3.10-006C6B?style=for-the-badge&color=3a3b3a&labelColor=%3a3b3a&logo=python&logoColor=FFFFFF"> <img src ='https://img.shields.io/github/repo-size/kvasik3000/Fire_Detection?style=for-the-badge&color=3a3b3a&labelColor=%3a3b3a&logo=weightsandbiases&logoColor=FFFFFF'> <img src = 'https://img.shields.io/github/commit-activity/m/kvasik3000/Fire_Detection?style=for-the-badge&color=3a3b3a&labelColor=%3a3b3a&logo=Alwaysdata&logoColor=FFFFFF'> <img src = 'https://img.shields.io/github/watchers/kvasik3000/Fire_Detection?style=for-the-badge&color=3a3b3a&labelColor=%3a3b3a&logo=exercism&logoColor=FFFFFF'>  <img src = 'https://img.shields.io/github/contributors/kvasik3000/Fire_Detection?style=for-the-badge&color=3a3b3a&labelColor=%3a3b3a&logo=teamspeak&logoColor=FFFFFF'>  

<details>
  <summary><h2>Контент</h2></summary>
  <ol>
    <li>
      <a>О проекте</a>
     <ul>
       <li><a href='#Проблема'> Проблема </a></li>
       <li><a href='#Задача'> Задача </a></li>
     </ul>
    </li>
    <li>
      <a href="#Решение"> Решение проекта </a>
      <ul>
        <li><a href="#Загрузка-проекта">Загрузка проекта</a></li>
       <li><a href='#Работа-с-проектом'>Работа с проектом</a></li> 
       <li><a href='#Оценка-модели'>Оценка модели</a></li> 
       <li><a href='#Результат-работы-модели'>Результат работы модели</a></li> 
      </ul>
    </li>
    <li><a href="#RoadMap">RoadMap</a></li>
    <li><a href="#Команда">Команда</a></li>
  </ol>
</details>

---

## Проблема

По данным МЧС России на 22 год в России было зафиксировано:
+  🔥 более 300 тысяч пожаров в черте города
+ 📈 количество погибших приближается к отметке в 8000, что составляет примерно 20 человек в день 
+ 🕑 время прибытия первого пожарного подразделения не превышает 10 минут, но для того, чтобы пожарный расчёт выехал, необходимо сообщение о пожаре
  
Данный проект предлагает решение проблемы предупреждения возгорания.

---

## Задача

Необходимо было разработать детектор огня и дыма, встраеваемый в камеры наружного и внутреннего видеонаблюдения, с целью предотвращения возгораний и пожаров.  

Для достижения наилучшего результата, разработка основывается на таких гарантах, как:

+ 🚩 Быстрое реагирование: Детектор позволит оперативно обнаруживать возгорания в режиме реального времени. Это даст возможность принять меры по локализации источника огня, тушению и эвакуации людей до того, как произойдет трагедия или убытки.

+ 🚩 Безопасность на производстве: В областях, где присутствуют горючие материалы, детектор огня поможет предотвратить взрывы, пожары и уберечь жизни сотрудников.

+ 🚩 Домашняя безопасность: В домах детектор огня обеспечит безопасность, обнаруживая возгорания от свечей, газовых плит, электроприборов и других источников.

+ 🚩 Экологическая защита: Детектор огня поможет предотвратить загрязнение окружающей среды и сохранить природные ресурсы.

+ 🚩 Экономические выгоды: Своевременное обнаружение возгорания снизит убытки, связанные с разрушением имущества и прерыванием производства.

---

## Решение

<img src="https://github.com/kvasik3000/Fire_Detection/blob/main/images/model.jpeg"  alt="1" width = 700px height = 360px > 

Нашим продуктом является нейросетевой алгоритм, состоящий из двух детекционных моделей Yolov8. Одна для детекции огня, вторая для дыма. 
Мы выделили дым, как отдельный признак возгорания, потому что по статистике МЧС более 60% смертей происходит от удушья, а не от высокой температуры. 
И в наше время - "время пластика", дым без огня весьма часто явление.

### Загрузка проекта

Рекомендованная версия Python 3.9
+ Клонируйте репозиторий нижеприведенной командой:
```
git clone https://github.com/kvasik3000/Fire_Detection
```
+ Откройте папку и установите необходимые библиотеки командой ниже:
```
pip install -r requirements.txt
```

### Работа с проектом 

Для запуска на изображении или на директории с изображениями используйте image_inference.py, который находится в директории inference

```bash
python .\inference\image_inference.py --images <path/to/image> -m .\models\garbage.onnx
```

Для запуска на видео или на директории с видео используйте video_inference.py, который находится в директории inference

```bash
python .\inference\video_inference.py --videos <path/to/video> -m .\models\garbage.onnx
```

Дополнительные параметры можно посмотреть, добавив --help.

```bash
python .\inference\image_inference.py --help
```
```bash
python .\inference\video_inference.py --help
```

### Оценка модели

Критерием оценивания качества работы продукта стало то, как хорошо она находит возгорание в кадре. Эти значения были получены на специальном датасете, который включается в себя максимально неудобные примеры для нашего алгоритма, с большим перекрытием или низким качеством изображения.  


| ✅ Recall по картинке      | ✅ Precision по картинке    |  ✅ mAP50                   |
|----------------------------|------------------------------|------------------------------|
|            95%             |               98%            |               85%            | 

Использование этих метрик обусловлено целью - максимально достоверно и точно найти возгорание. А потом уже пользуясь информацией о величине очага, можно судить о серьезности проблемы.

### Результат работы модели

<img src="https://github.com/kvasik3000/Fire_Detection/blob/main/images/result.gif"  alt="1" width = 700px height = 360px > 

---

## RoadMap

- [ ] ⭕ Оценка задачи
    - [x] Изучение и уточнение требований
    - [x] Анализ готовых решений
- [ ] ⭕ Сбор датасета
    - [x] Поиск готовых датасетов
    - [x] Поиск фото через поисковые системы
    - [x] Предобработка изображений
- [ ] ⭕ Обучение общей модели
    - [x] Обучение модели, которая детектирует огонь и дым одновременно
- [ ] ⭕ Обучение двух моделей
    - [x] Разделение задачи на две: обучение модели огня, обучение модели дыма

---
 
## Команда:
Наша команда состоит из 5 человек. У нас есть опыт побед в различных соревнованиях и хакатонах по машинному обучению и не только.
+ Scrum master 📝: [Квас Андрей](https://github.com/kvasik3000)
+ ML 💻: [Володина Софья](https://github.com/PiroJOJO) , [Бабенко Егор](https://github.com/JooudDoo)
+ ML-Ops 🛠: [Лейсле Александр](https://github.com/HerrPhoton)
+ QA 🔫: [Шитенко Алина](https://github.com/alincnl)


