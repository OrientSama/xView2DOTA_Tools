import os
from pathlib import Path

"""
用来将DOTA数据集与XVIEW数据集的公共类筛选出来
按照论文先比较三类
DOTA:   {plane ship storage-tank} container-crane helicopter
xView:  {plane: [fixed-wing-aircraft, small-aircraft, cargo-plane]
         ship: [container-ship motorboat sailboat tugboat fishing-vessel maritime-vessel barge ferry yacht oil-tanker] 
         storage-tank} container-crane helicopter
xveiw classes
  0: Fixed-wing-Aircraft
  1: Small-Aircraft
  2: Cargo-Plane
  3: Helicopter
  4: Passenger-Vehicle
  5: Small-Car
  6: Bus
  7: Pickup-Truck
  8: Utility-Truck
  9: Truck
  10: Cargo-Truck
  11: Truck-w/Box
  12: Truck-Tractor
  13: Trailer
  14: Truck-w/Flatbed
  15: Truck-w/Liquid
  16: Crane-Truck
  17: Railway-Vehicle
  18: Passenger-Car
  19: Cargo-Car
  20: Flat-Car
  21: Tank-car
  22: Locomotive
  23: Maritime-Vessel
  24: Motorboat
  25: Sailboat
  26: Tugboat
  27: Barge
  28: Fishing-Vessel
  29: Ferry
  30: Yacht
  31: Container-Ship
  32: Oil-Tanker
  33: Engineering-Vehicle
  34: Tower-crane
  35: Container-Crane
  36: Reach-Stacker
  37: Straddle-Carrier
  38: Mobile-Crane
  39: Dump-Truck
  40: Haul-Truck
  41: Scraper/Tractor
  42: Front-loader/Bulldozer
  43: Excavator
  44: Cement-Mixer
  45: Ground-Grader
  46: Hut/Tent
  47: Shed
  48: Building
  49: Aircraft-Hangar
  50: Damaged-Building
  51: Facility
  52: Construction-Site
  53: Vehicle-Lot
  54: Helipad
  55: Storage-Tank
  56: Shipping-container-lot
  57: Shipping-Container
  58: Pylon
  59: Tower
"""

# 修改了类别的map
fixed_category = {'fixed-wing-aircraft': 'plane', 'plane': 'plane', 'small-aircraft': 'plane', 'cargo-plane': 'plane',
                  'storage-tank': 'storage-tank',
                  'ship': 'ship', 'container-ship': 'ship', "motorboat": "ship",
                  "sailboat": "ship", "tugboat": "ship", "fishing-vessel": "ship", "maritime-vessel": "ship",
                  "barge": "ship", 'ferry': "ship", 'yacht': "ship", 'oil-tanker': "ship"}


def common_category(path: Path):
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
                    if line.startswith(('image', 'gsd')):
                        continue
                    category = line.split(' ')[-2]
                    if category in fixed_category:
                        flag = True
                        with open(common_ann/id, 'a') as common_f:
                            common_f.write(line.replace(category, fixed_category[category]))
                if flag and (path_parent / 'images' / (new_id + '.disabled')).exists():
                    os.rename(path_parent / 'images' / (new_id + '.disabled'), path_parent / 'images' / new_id)
                if not flag and (path_parent / 'images' / new_id).exists():
                    os.rename(path_parent / 'images' / new_id, path_parent / 'images' / (new_id + '.disabled'))


split = Path("E:/Dataset/Split")
dota_train = split / 'dota_train'
# dota_val = split / 'dota_val'
# xview = split / 'xview_train'
for p in [dota_train, ]:
    common_category(p / 'annfiles')
