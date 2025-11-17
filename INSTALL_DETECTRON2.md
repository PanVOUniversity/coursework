# Инструкция по установке Detectron2

## Проблема с установкой на Windows

При установке Detectron2 через `pip install 'git+https://...'` может возникнуть ошибка:
```
ModuleNotFoundError: No module named 'torch'
```

Это происходит потому, что pip создает изолированное окружение для сборки, где PyTorch недоступен.

## Решение 1: Установка через git clone (рекомендуется)

Этот метод работает надежнее на Windows:

```bash
# 1. Убедитесь, что PyTorch установлен
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 2. Клонируйте репозиторий
git clone https://github.com/facebookresearch/detectron2.git

# 3. Перейдите в директорию и установите
cd detectron2
pip install -e .

# 4. Вернитесь обратно
cd ..
```

## Решение 2: Установка предсобранных wheel-файлов

Для PyTorch 2.9 попробуйте wheel для torch2.0 (обычно совместимы):

```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

Если не работает, попробуйте другие версии:
```bash
# PyTorch 2.1
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html

# PyTorch 1.13
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.13/index.html
```

## Решение 3: Установка PyTorch в build environment (продвинутый метод)

Если нужно использовать `pip install git+...`, можно попробовать установить PyTorch в build environment:

```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

Но этот метод может не работать на всех системах.

## Проверка установки

После установки проверьте:

```bash
python -c "import detectron2; print(f'Detectron2 {detectron2.__version__}')"
```

## Альтернативы

Если Detectron2 не устанавливается, можно использовать альтернативные модели:

1. **U-Net** из `segmentation_models_pytorch`:
   ```bash
   pip install segmentation-models-pytorch
   ```

2. **Mask2Former**:
   ```bash
   pip install mask2former
   ```

3. **SAM (Segment Anything Model)**:
   ```bash
   pip install segment-anything
   ```

## Дополнительная информация

- Официальная документация: https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md
- Проблемы с установкой: https://github.com/facebookresearch/detectron2/issues

