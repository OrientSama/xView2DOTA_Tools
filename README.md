### 该文件夹内的python文件主要用于xView2DOTA的数据集处理
### The Python files in this folder are mainly used for dataset processing in xView2DOTA
```
TOOLS
│  Active.py            # 用来去除文件名 '.disabled' 后缀
│  DOTA2COCO.py         # 将DOTA格式转为COCO格式
│  dota_utils.py
│  dota_xview.py        # 将xView与DOTA公共类提取出来
│  splitdataset.py      # 将有标签的训练集重新划分
│  xView.py             # 将xView的标签转为DOTA格式
│
├─dota
│  │  README.md
│  │
│  └─split
│      │  img_split.py              # 分割图片
│      │
│      └─split_configs              # 配置文件
│              cls_trainval.json
│              ms_test.json
│              ms_train.json
│              ms_trainval.json
│              ms_val.json
│              ss_test.json
│              ss_train.json
│              ss_trainval.json
│              ss_val.json
│              xview.json
```
