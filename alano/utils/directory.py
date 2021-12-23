import os
import shutil


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_directory_hard(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)
