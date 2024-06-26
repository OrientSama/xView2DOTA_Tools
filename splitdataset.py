import os
from pathlib import Path
from shutil import move, rmtree
import random


def mk_file(file_path: str | Path):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)
    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.8
    data_root = Path(r"E:\Dataset\Split\dota_train")
    origin_path = data_root / "common_ann"
    assert os.path.exists(origin_path), "path '{}' does not exist.".format(origin_path)

    ann_list = os.listdir(origin_path)
    # 建立保存val集的文件夹
    val_root = data_root.parent / "dota_val"
    val_ann = val_root / "common_ann"
    val_img = val_root / "images"

    mk_file(val_ann)
    mk_file(val_img)

    num = len(ann_list)
    # 随机采样验证集的索引
    eval_index = random.sample(ann_list, k=int(num * split_rate))

    for index, ann in enumerate(ann_list):
        if ann in eval_index:
            # 将分配至验证集中的文件复制到相应目录
            move(origin_path / ann, val_ann / ann)
            move(data_root / "images" / ann.replace(".txt", ".png"), val_img / ann.replace(".txt", ".png"))
        print("\r processing [{}/{}]".format(index+1, num), end="")  # processing bar

    print("processing done!")


if __name__ == '__main__':
    main()