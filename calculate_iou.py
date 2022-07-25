from configparser import ConfigParser

import numpy as np
from cv2 import cv2
import pandas as pd
import os
import cv2
import numpy as np
import glob
import mlflow

# class_names = ["infiltration", "nodule", "consolidation", "fibrosis", "pleural_thickening"]


def iou_coef(y_true, y_pred, smooth=1):
  intersection = np.sum(np.abs(y_true * y_pred), axis=(1,2))
  union = np.sum(y_true,(1,2)) + np.sum(y_pred, (1,2)) - intersection
  iou = np.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def dice_coef(y_true, y_pred, smooth=1):
  intersection = np.sum(y_true * y_pred, axis=(1,2))
  union = np.sum(y_true, axis=(1,2)) + np.sum(y_pred, axis=(1,2))
  dice = np.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice


config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
class_names = cp["DEFAULT"].get("class_names").split(",")

# CAM config
run_id = mlflow.list_run_infos("0")[0].run_id
# default config
output_dir_root = cp["DEFAULT"].get("output_dir")
output_dir = output_dir_root + run_id

for class_name in class_names:
    path_gt = f"data/vindr-cxr/data_for_iou/{class_name}"
    # path_gc = f"experiments_our_model/20/cam/{class_name}"
    path_gc = f"{output_dir}/cam/{class_name}"
    # folder_path = f"data_for_iou\\{class_name}\\"

    files_gc = glob.glob(path_gc+"/*")
    files_gt = []
    for name in files_gc:
        files_gt.append(path_gt + "/" + name.split(".")[-2] + "." + name.split(".")[-1])
    # files_gt = glob.glob(path_gt+"\\*")
    # for name in files_gt:
    #     if name in files_gc
    y_pred = np.array([np.where(cv2.imread(img, 0) > 0, 1, 0) for img in files_gc], dtype=np.float32)
    y_true = np.array([np.where(cv2.imread(img, 0) > 0, 1, 0) for img in files_gt], dtype=np.float32)

    print(class_name)
    print("IOU: ", iou_coef(y_true, y_pred))
    print("Dice: ", dice_coef(y_true, y_pred))


def make_masks():
    img = cv2.imread("")
    data = pd.read_csv("data_for_iou\\BBOX.csv")
    for class_name in class_names:
        df = data.loc[data["Finding Label"] == class_name]
        for index, row in df.iterrows():
            name, _, x, y, w, h = row
            folder_path = f"data_for_iou\\{class_name}"
            file_path = f"data_for_iou\\{class_name}\\{name}"
            if not os.path.exists(folder_path):
                os.makedirs(f"data_for_iou\\{class_name}")
            if os.path.isfile(file_path):
                mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask_img = np.zeros((1024, 1024))
            # points = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
            # points = np.array(points, dtype=np.int32)
            # # reshape points to infer # of rows
            # points = points.reshape((-1, 1, 2))
            x1, y1, x2, y2 = np.array([x, y, x+w, y+h], dtype=np.int32)
            mask_img = cv2.rectangle(mask_img, (x1, y1), (x2, y2), (255), cv2.FILLED)
            cv2.imwrite(file_path, mask_img)

            # mask_img = data["Finding Label"]
