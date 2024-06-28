### 该文件夹内的python文件主要用于xView2DOTA的数据集处理
### The Python files in this folder are mainly used for dataset processing in xView2DOTA

#### 文件夹结构

```
TOOLS
│  Active.py            # 用来去除文件名 '.disabled' 后缀
│  DOTA2COCO.py         # 将DOTA格式转为COCO格式
│  dota_utils.py
│  dota_xview.py        # 将xView与DOTA公共类提取出来
│  splitdataset.py      # 将有标签的训练集重新划分
│  xView.py             # 将xView的标签转为DOTA格式
|  xView.yaml           # 名称、编号对应文件
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
#### 使用说明
- 首先将xView数据的标签转化成DOTA格式 [xView.py](xView.py)
- 修改`./Tools/dota/split/splilt_configs/`文件夹下的配置文件`ss_train.json/xview.json等等`（只需要改里面的路径）
- 将xView数据集，DOTA数据集进行分割 [img_split.py](dota%2Fsplit%2Fimg_split.py)
- 然后对分割后的图片进行公共类的提取 [dota_xview.py](dota_xview.py)
- 提取完成后将标签转化成COCO格式 [DOTA2COCO.py](DOTA2COCO.py)