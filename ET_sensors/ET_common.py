import os


def check_dir(path1, name_dir):
    new_path = path1 + name_dir
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path
