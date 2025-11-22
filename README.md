# Frame Segmentation Pipeline

Готовый к запуску репозиторий для генерации синтетических веб-страниц с фреймами, снятия полноразмерных скриншотов через Playwright+Chromium, генерации попиксельных instance-масок, конвертации в COCO, обучения Mask R-CNN (Detectron2), инференса и вычисления сдвигов для устранения наложений.

## Структура проекта

```
frame-segmentation/
├── README.md                    # Инструкции по запуску
├── requirements.txt             # Python зависимости
├── Dockerfile                   # Docker образ с GPU/CPU поддержкой
├── data-generation/
│   ├── html_generator.py        # Генератор HTML страниц с фреймами
│   ├── playwright_render.py     # Скриншоты через Playwright
│   ├── make_masks.py            # Генерация instance-масок
│   └── coco_converter.py         # Конвертация в COCO формат
├── detectron/
│   ├── train.py                 # Обучение Mask R-CNN
│   └── infer_and_postprocess.py # Инференс и вычисление сдвигов
├── utils/
│   ├── geometry.py              # Функции геометрии (сдвиги, bbox)
│   └── color_mapping.py         # id↔color преобразования
├── tests/
│   └── test_geometry.py         # Unit-тесты для geometry
├── notebooks/
│   └── demo_inference.ipynb     # Jupyter notebook с демо
└── examples/
    └── run_all.sh               # Скрипт запуска всего пайплайна
```

## Установка

### Локальная установка

1. **Установите Python 3.10+**
2. **Установите зависимости:**

```bash
pip install -r requirements.txt
```

3. **Установите Playwright и браузер:**

```bash
playwright install chromium
```

4. **Установите PyTorch (обязательно перед Detectron2):**

**Важно:** Detectron2 требует PyTorch. Сначала установите PyTorch, затем Detectron2.

Для CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Для GPU (CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Для GPU (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Проверьте установку:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

5. **Установите Detectron2:**

**Важно:** Detectron2 требует компиляции из исходников или установки предварительно собранных wheel-файлов.

**⚠️ На Windows установка через `pip install git+...` часто не работает из-за изолированного окружения сборки!**

**Метод 1: Установка через git clone (РЕКОМЕНДУЕТСЯ для Windows):**

**ВАЖНО:** Убедитесь, что PyTorch уже установлен перед этим шагом!

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

Или в одну строку (PowerShell):

```powershell
git clone https://github.com/facebookresearch/detectron2.git; cd detectron2; pip install -e .; cd ..
```

**Метод 1b: Установка через pip (может не работать на Windows):**

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Если видите ошибку "No module named 'torch'", используйте Метод 1.

**Подробные инструкции:** См. файл `INSTALL_DETECTRON2.md` в корне проекта.

**Метод 2: Установка предварительно собранных wheel-файлов:**

Сначала проверьте версию PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

Затем попробуйте установить соответствующий wheel (для PyTorch 2.9 попробуйте torch2.0):

```bash
# Для PyTorch 2.0+ и CPU (совместимо с PyTorch 2.9)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html

# Если не работает, попробуйте torch2.1 (если доступен)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html

# Для PyTorch 1.13 и CPU (может не работать на новых системах)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.13/index.html

# Для GPU версий проверьте совместимость на https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md
```

**Примечание:** Если wheel-файлы не найдены для вашей версии PyTorch, используйте Метод 1 (git clone).

**Метод 3: Если возникают проблемы, используйте альтернативу:**

Если Detectron2 не устанавливается, можно использовать альтернативные модели:

- **U-Net** из `segmentation_models_pytorch`
- **Mask2Former**
- **SAM (Segment Anything Model)**

Пример с U-Net:

```bash
pip install segmentation-models-pytorch
```

### Docker установка

**CPU версия:**

```bash
docker build -t frame-seg .
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs frame-seg
```

**GPU версия:**

```bash
docker build --build-arg CUDA=1 -t frame-seg .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs frame-seg
```

## Использование

### Быстрый старт

Запустите весь пайплайн одной командой:

```bash
./examples/run_all.sh --n 100
```

Пропустить обучение (для быстрой проверки):

```bash
./examples/run_all.sh --n 50 --skip-train
```

### Пошаговое выполнение

1. **Генерация HTML страниц:**

```bash
# Стандартный размер
python data-generation/html_generator.py --n 100 --min-frames 3 --max-frames 10

```

2. **Рендеринг скриншотов:**

```bash
# Viewport автоматически читается из метаданных (page_width, page_height)
python data-generation/playwright_render.py --input-dir data/pages --output-dir data/screenshots --workers 1

# Если метаданных нет, можно задать дефолтный viewport
python data-generation/playwright_render.py --viewport-width 2560 --viewport-height 1440
```

3. **Генерация масок:**

```bash

```

4. **Конвертация в COCO:**

```bash
python data-generation/coco_converter.py --mask-dir data/masks --meta-dir data/meta --output-dir data/coco
```

5. **Обучение модели:**

```bash
python detectron/train.py --coco-dir data/coco --output-dir outputs --epochs 100 --batch-size 2 --gpu
```

6. **Инференс и постобработка:**

```bash
python detectron/infer_and_postprocess.py --weights outputs/model_final.pth --input-dir data/screenshots --output-dir data/results --gpu
```

## Детали реализации

### HTML генератор

- Генерирует случайные `div.frame` с `position: absolute/fixed/relative`
- Случайные размеры, позиции, z-index, фон, тени
- **Поддержка закругленных углов** (`border-radius`)
- Lorem ipsum текст и случайные изображения через Picsum Photos
- Сохраняет метаданные в JSON (x, y, w, h, z_index, id, border_radius)

### Формат метаданных JSON

Метаданные сохраняются в файлы `data/meta/page_*.json` со следующей структурой:

```json
{
  "page_id": int,                    // ID страницы
  "page_width": int,                 // Ширина viewport, использованная при рендеринге
  "page_height": int,                // Высота страницы
  "header": {                         // Конфигурация header (опционально)
    "id": "header",
    "x": int,                         // Позиция X
    "y": int,                         // Позиция Y
    "w": int,                         // Ширина
    "h": int,                         // Высота
    "z_index": int,                   // Z-index для наложения
    "border_radius": int,             // Радиус скругления углов
    "position": str,                  // CSS position (обычно "relative")
    "bg_color": str,                  // Цвет фона (CSS формат)
    "box_shadow": str,                // CSS box-shadow
    "is_header": true                 // Флаг идентификации header
  },
  "footer": {                         // Конфигурация footer (опционально)
    // Аналогичная структура как у header
    "is_footer": true
  },
  "sliders": [                        // Массив слайдеров с фонами
    {
      "id": int,                      // ID слайдера
      "height": int,                   // Высота слайдера
      "top": int,                      // Позиция сверху
      "background": {
        "type": str,                  // Тип фона (например, "linear_gradient")
        "css": str                    // CSS строка для фона
      }
    }
  ],
  "frames": [                         // Массив фреймов
    {
      "id": int,                      // Уникальный ID фрейма
      "x": int,                       // Позиция X (абсолютная)
      "y": int,                       // Позиция Y (абсолютная, с учетом header)
      "w": int,                       // Ширина фрейма
      "h": int,                       // Высота фрейма
      "z_index": int,                 // Z-index для наложения (1-999)
      "border_radius": int,           // Радиус скругления углов (0 до min(w,h)/4)
      "bg_color": str,                // Цвет фона (CSS формат, например "rgb(241, 239, 212)")
      "box_shadow": str,              // CSS box-shadow
      "in_header": bool,              // Флаг: находится ли фрейм в области header
      "in_footer": bool               // Флаг: находится ли фрейм в области footer
    }
  ]
}
```

**Примечания:	**

- Все координаты и размеры указаны в пикселях
- `page_width` соответствует размеру viewport, использованному при рендеринге скриншота
- Фреймы имеют координаты относительно начала страницы, но с учетом высоты header (y-координата сдвинута на высоту header)
- `z_index` фреймов ограничен диапазоном 1-999 (header и footer имеют z-index 1000)
- `border_radius` может быть 0 (прямоугольник) или положительным числом (скругленные углы)

### Playwright рендерер

- Headless Chromium для рендеринга
- **Viewport автоматически читается из метаданных** (page_width, page_height из JSON)
- Если метаданных нет, используются дефолтные значения (1920x1080) или заданные через `--viewport-width/height`
- Отключает анимации через JS инъекцию
- Прокрутка до конца страницы с ожиданием network idle
- Полноразмерные скриншоты (`fullPage`)
- Поддержка параллельного рендеринга (async)

### Генератор масок

- Создает попиксельные instance-маски с цветовым кодированием ID
- **Корректно обрабатывает закругленные углы** через `draw_rounded_rectangle_mask()`
- Фон = (0,0,0), инстансы имеют уникальные RGB цвета
- Сохраняет маски в PNG формате

### COCO конвертер

- Извлекает контуры из instance-масок через `cv2.findContours`
- **Контуры автоматически учитывают закругленные углы**
- Конвертирует в RLE формат через pycocotools
- Сохраняет z_index и border_radius в аннотациях
- Разделяет на train/val (80/20)

### Обучение Detectron2

- Регистрация датасета через `DatasetCatalog`
- Использует конфиг `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`
- Настраивает `ROI_HEADS.NUM_CLASSES = 1` (только "frame")
- Поддержка CPU и GPU
- Сохраняет чекпойнты в `outputs/`

### Инференс и постобработка

- Загружает обученную модель
- Предсказывает маски и bbox на скриншотах
- Вычисляет перекрытия между парами масок
- Определяет "кто сверху" (по z_index или покрытию)
- Вычисляет минимальные сдвиги для устранения наложений
- Сохраняет результаты в JSON и визуализацию `overlap_mask.png`

## Выходные данные

- `data/pages/*.html` - Сгенерированные HTML страницы
- `data/meta/*.json` - Метаданные фреймов
- `data/screenshots/*.png` - Полноразмерные скриншоты
- `data/masks/*_instance_mask.png` - Instance-маски
- `data/coco/annotations/instances_{train,val}.json` - COCO аннотации
- `data/coco/{train,val}/*.png` - Изображения для COCO
- `outputs/model_final.pth` - Обученная модель
- `data/results/page_*.json` - Результаты инференса с перекрытиями и сдвигами
- `data/results/page_*_overlap_mask.png` - Визуализация перекрытий

## Тестирование

Запустите unit-тесты:

```bash
python -m pytest tests/
```

## Jupyter Notebook

Откройте `notebooks/demo_inference.ipynb` для интерактивной визуализации результатов.

## Альтернатива Detectron2

Если Detectron2 вызывает сложности при установке, можно использовать альтернативные модели сегментации:

- U-Net из `segmentation_models_pytorch`
- Mask2Former
- SAM (Segment Anything Model)

Примеры интеграции можно найти в документации соответствующих библиотек.

## Требования

- Python 3.10+
- Playwright с Chromium
- Detectron2 (или альтернатива)
- CUDA (опционально, для GPU)

## Лицензия

MIT

## Автор

Создано для курсовой работы по компьютерному зрению.
