#!/usr/bin/env python3
"""
Скрипт для очистки всех файлов из всех папок в директории data.
Удаляет все файлы из подпапок, но сохраняет структуру папок.
"""

import os
import sys
from pathlib import Path


def clear_data_folders(data_dir="data"):
    """
    Очищает все файлы из всех подпапок в указанной директории.
    
    Args:
        data_dir: Путь к директории data (по умолчанию "data")
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Ошибка: директория '{data_dir}' не существует.")
        return False
    
    if not data_path.is_dir():
        print(f"Ошибка: '{data_dir}' не является директорией.")
        return False
    
    deleted_count = 0
    deleted_files = []
    
    # Проходим по всем подпапкам в data
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            print(f"Очистка папки: {subdir.name}")
            
            # Удаляем все файлы в подпапке
            for file_path in subdir.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        deleted_files.append(str(file_path))
                        print(f"  Удален: {file_path.name}")
                    except Exception as e:
                        print(f"  Ошибка при удалении {file_path.name}: {e}")
    
    print(f"\nГотово! Удалено файлов: {deleted_count}")
    
    if deleted_files:
        print("\nУдаленные файлы:")
        for file_path in deleted_files:
            print(f"  - {file_path}")
    
    return True


if __name__ == "__main__":
    # Можно указать путь к data как аргумент командной строки
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    print(f"Очистка всех файлов из папок в '{data_directory}'...")
    print("-" * 50)
    
    success = clear_data_folders(data_directory)
    
    if success:
        print("\nОчистка завершена успешно!")
    else:
        print("\nОчистка завершена с ошибками.")
        sys.exit(1)

