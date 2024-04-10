from pathlib import Path
import os

# 用来去除文件名 '.disabled' 后缀

def remove_disabled_ext(path):
    file_list = os.listdir(path)
    for id in file_list:
        if id.endswith('.disabled'):
            os.rename(path / id, path / id.replace('.disabled', ''))

dota = Path('E:/Dataset/DOTA1.5')
train_image = dota / 'train' / 'images'
val_image = dota / 'val' / 'images'
train_ann = dota / 'train' / 'labelTxt-v1.5'
val_ann = dota / 'val' / 'labelTxt-v1.5'
for p in [train_image, train_ann, val_image, val_ann]:
    remove_disabled_ext(p)