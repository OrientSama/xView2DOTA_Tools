import json
import os, sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml, torch

# from utils.dataloaders import autosplit
# from utils.general import download, xyxy2xywhn

# def xyxy2xywh(x):
#     """Convert nx4 boxes from [x1, y1, x2, y2] to DOTA style [x1, y1, x2, y1, x2, y2, x1, y2] """
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
#     y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
#     y[..., 2] = x[..., 2] - x[..., 0]  # width
#     y[..., 3] = x[..., 3] - x[..., 1]  # height
#     return y


def convert_boxes_to_dota_style(boxes):
    """Convert nx4 boxes from [x1, y1, x2, y2] to DOTA style [x1, y1, x2, y1, x2, y2, x1, y2] """
    dota_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        dota_box = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
        dota_boxes.append(dota_box)

    return np.array(dota_boxes)


def convert_labels(fname=Path('xView/xView_train.geojson')):
    # Convert xView geoJSON labels to YOLO format
    path = fname.parent
    with open(fname) as f:
        print(f'Loading {fname}...')
        data = json.load(f)

    # Make dirs

    labels = Path(path / 'labels' / 'train')
    os.system(f'rm -rf {labels}') if sys.platform.startswith('linux') else os.system(f"rmdir /Q /S {labels}")
    labels.mkdir(parents=True, exist_ok=True)

    # xView classes 11-94 to 0-59
    xview_class2index = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
                         12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
                         29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46,
                         47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59]
    # 从文件中加载 YAML 数据
    with open(r"E:\PyCharm_Projects\Domain_Adaption\yolov5\data\xView.yaml", 'r', encoding='utf-8') as file:
        xview_data = yaml.load(file, Loader=yaml.SafeLoader)
        xview_index2name = xview_data['names']

    shapes = {}
    for feature in tqdm(data['features'], desc=f'Converting {fname}'):
        p = feature['properties']
        if p['bounds_imcoords']:
            id = p['image_id']
            file = path / 'images' / 'train' / id
            if file.exists():  # 1395.tif missing
                try:
                    box = np.array([int(num) for num in p['bounds_imcoords'].split(",")])
                    assert box.shape[0] == 4, f'incorrect box shape {box.shape[0]}'
                    cls = p['type_id']
                    cls = xview_class2index[int(cls)]  # xView class to 0-60
                    assert 59 >= cls >= 0, f'incorrect class index {cls}'
                    assert xview_index2name, f'xview.yaml not Found!'
                    name = xview_index2name[cls]

                    # Write  label
                    if id not in shapes:
                        shapes[id] = Image.open(file).size
                    box = convert_boxes_to_dota_style(box[None].astype(float))
                    with open((labels / id).with_suffix('.txt'), 'a') as f:
                        f.write(f"{' '.join(f'{x:.1f}' for x in box[0])} {name.lower()} 0\n")  # write label.txt
                except Exception as e:
                    print(f'WARNING: skipping one label for {file}: {e}')


# Download manually from https://challenge.xviewdataset.org
dir = Path("E:/Dataset/xView")  # dataset root dir
# urls = ['https://d307kc0mrhucc3.cloudfront.net/train_labels.zip',  # train labels
#         'https://d307kc0mrhucc3.cloudfront.net/train_images.zip',  # 15G, 847 train images
#         'https://d307kc0mrhucc3.cloudfront.net/val_images.zip']  # 5G, 282 val images (no labels)
# download(urls, dir=dir, delete=False)

# Convert labels
convert_labels(dir / 'xView_train.geojson')

# Move images
images = Path(dir / 'images')
images.mkdir(parents=True, exist_ok=True)
try:
    Path(dir / 'train_images').rename(dir / 'images' / 'train')
    Path(dir / 'val_images').rename(dir / 'images' / 'val')
except Exception as e:
    print(e)

# Split
# autosplit(dir / 'images' / 'train')
