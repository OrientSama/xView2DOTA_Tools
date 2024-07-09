"""
the summarize function is copy from
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_object_detection/faster_rcnn/validation.py
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json
import argparse
from pathlib import Path
import torch
import numpy as np


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self['params']
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            """
            T           = len(p.iouThrs)   # [0.5, 0.55, ... ,0.95]   10
            R           = len(p.recThrs)   # [0, 0.01, 0.02, ..., 1]  101
            K           = len(p.catIds) if p.useCats else 1
            A           = len(p.areaRng)   # [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
            M           = len(p.maxDets)   # [1, 10, 100]
            """
            s = self['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self['params'].maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self['params'].maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self['params'].maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self['params'].maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self['params'].maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self['params'].maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self['params'].maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self['params'].maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self['params'].maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self['params'].maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self['params'].maxDets[2])

    print_info = "\n".join(print_list)

    if not self:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(args):
    assert os.path.exists(args.path)
    assert os.path.exists(args.anno)
    # read class_indict
    label_json_path = args.anno
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {_dict['id']: _dict['name'] for _dict in class_dict['categories']}

    coco_eval = torch.load(args.path, map_location='cpu')
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    mAP_50_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        mAP_50_list.append(" {:15}: {}".format(category_index[i], stats[1]))

    print_voc = "\n".join(mAP_50_list)
    print(print_voc)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    # 将验证结果保存至txt文件中
    with open(output_dir / "record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)

    # you can save this file by 'torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)'
    parser.add_argument('--path', default='', help='the path of eval.pth')
    parser.add_argument('--anno', default='', help="the path of coco-style annotations")
    parser.add_argument('--output_dir', default='', help="output directory")
    _args = parser.parse_args()
    main(_args)
