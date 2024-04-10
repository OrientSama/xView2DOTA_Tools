import os
from pathlib import Path

"""
用来将DOTA数据集与XVIEW数据集的公共类筛选出来
按照论文先比较三类
DOTA:   {plane ship storage-tank} container-crane helicopter
xView:  {plane: [fixed-wing-aircraft, small-aircraft, cargo-plane] container-ship storage-tank} container-crane helicopter
"""

fixed_category = {'plane': 'plane', 'ship': 'ship', 'storage-tank': 'storage-tank', 'fixed-wing-aircraft': 'plane',
                  'small-aircraft': 'plane', 'cargo-plane': 'plane', 'container-ship': 'ship'}


def common_cat(path: Path):
    file_list = os.listdir(path)
    path_parent = path.parent
    common_ann = path_parent / "common_ann"
    common_ann.mkdir(parents=True, exist_ok=True)
    for id in file_list:
        new_id = id.replace('.txt', '.png')
        with open(path / id, 'r') as f:
            data = f.readlines()
            if len(data) == 0:
                if (path_parent / 'images' / new_id).exists():
                    os.rename(path_parent / 'images' / new_id, path_parent / 'images' / (new_id + '.disabled'))
            else:
                flag = False  # 用于标记是否存在三类物体
                for line in data:
                    category = line.split(' ')[-2]
                    if category in fixed_category:
                        flag = True
                        with open(common_ann/id, 'a') as common_f:
                            common_f.write(line.replace(category, fixed_category[category]))
                if not flag and (path_parent / 'images' / new_id).exists():
                    os.rename(path_parent / 'images' / new_id, path_parent / 'images' / (new_id + '.disabled'))


split = Path("E:/Dataset/Split")
dota = split / 'dota_trainval'
xview = split / 'xview_train'
for p in [dota, xview]:
    common_cat(p / 'annfiles')
