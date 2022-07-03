import cv2
import pydicom as dcm
import numpy as np
import os
import pandas as pd



def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, dataset))
    total_count = df.shape[0]
    labels = df[class_names].values
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts


def dicom_to_pngs(data_type):
    annotations_path = f"./data/vindr-cxr/annotations/annotations_{data_type}_5.csv"
    image_root_path = f"./data/vindr-cxr/{data_type}/"
    files = pd.read_csv(annotations_path)
    filtered_image_ids = files["image_id"].unique().tolist()
    for image_id in filtered_image_ids:
        image_name = image_id + ".dicom"
        ds = dcm.dcmread(os.path.join(image_root_path, image_name))
        img = ds.pixel_array
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            img = np.amax(img) - img
        img = img - np.min(img)
        mask = make_annotation_mask_from_bbox((img.shape[0], img.shape[1], 3), files.loc[files["image_id"] == image_id])
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        img = ((np.maximum(img, 0) / img.max()) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(f"./data/vindr-cxr/{data_type}_pngs", image_name.replace(".dicom", ".png")), img)
        cv2.imwrite(os.path.join(f"./data/vindr-cxr/{data_type}_masks", image_name.replace(".dicom", "_MASK.png")),
                    mask)


def dicom_to_pngs_unchanged(data_type):
    annotations_path = f"./data/vindr-cxr/annotations/annotations_{data_type}_5.csv"
    image_root_path = f"./data/vindr-cxr/{data_type}/"
    files = pd.read_csv(annotations_path)
    filtered_image_ids = files["image_id"].unique().tolist()
    for image_id in filtered_image_ids:
        image_name = image_id + ".dicom"
        ds = dcm.dcmread(os.path.join(image_root_path, image_name))
        img = ds.pixel_array
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            img = np.amax(img) - img
        img = img - np.min(img)
        img = ((np.maximum(img, 0) / img.max()) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(f"./data/vindr-cxr/images_unchanged", image_name.replace(".dicom", ".png")), img)


def preprocess_classification_records(data_type):
    df = pd.read_csv(f"./data/vindr-cxr/annotations/image_labels_{data_type}.csv")
    # Filtered 5 diseases for our use
    if data_type == "train":
        df = df[["image_id", "rad_id", "Consolidation", "Nodule/Mass", "Pleural thickening", "Infiltration",
             "Pulmonary fibrosis"]]
    else:
        df = df[["image_id", "Consolidation", "Nodule/Mass", "Pleural thickening", "Infiltration",
                 "Pulmonary fibrosis"]]
    # Filtered patients with any disease (remove healthy patients)
    df = df[df.isin([1]).any(axis=1)]
    # Added all the classification of different radiologists
    df = df.groupby("image_id").sum()
    # Replaced greater values with 1
    df = df.where(df == 0, other=1)
    df.to_csv(f"./data/vindr-cxr/annotations/image_labels_{data_type}_5.csv")


def preprocess_annotation_records(data_type):
    df = pd.read_csv(f"./data/vindr-cxr/annotations/annotations_{data_type}.csv")
    disease_names = ["Consolidation", "Nodule/Mass", "Pleural thickening", "Infiltration",
                     "Pulmonary fibrosis"]
    df = df.loc[df['class_name'].isin(disease_names)]
    df.to_csv(f"./data/vindr-cxr/annotations/annotations_{data_type}_5.csv")


def make_annotation_mask_from_bbox(image_size, data_rows):
    mask = np.zeros(image_size, dtype=np.uint8)
    mask[:, :, 2] = 1
    for index, row in data_rows.iterrows():
        mask = cv2.rectangle(mask, (int(row["x_min"]), int(row["y_min"])), (int(row["x_max"]), int(row["y_max"])),
                             (1, 0, 1), cv2.FILLED)
    return mask


def make_csv_for_model(data_type):
    df = pd.read_csv(f"./data/vindr-cxr/annotations/{data_type}.csv")
    df['mask_image_id'] = df['image_id'] + "_MASK.png"
    df['image_id'] += ".png"
    cols = list(df.columns)
    cols = cols[:1:] + cols[6::] + cols[1:6:]
    df = df[cols]
    df.to_csv(f"./data/vindr-cxr/annotations/{data_type}.csv")


def train_test_split():
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-multilearn"])
    from skmultilearn.model_selection import iterative_train_test_split
    disease_names = ["Consolidation", "Nodule/Mass", "Pleural thickening", "Infiltration",
     "Pulmonary fibrosis"]
    df = pd.read_csv(f"./data/vindr-cxr/annotations/train.csv")
    X_train, y_train, X_val, y_val = iterative_train_test_split(df.iloc[:, 1:3].to_numpy(), df.iloc[:, 3:].to_numpy(), test_size=0.15)
    # df.iloc[:, 1:].sum()/iterative_train_test_split(df.iloc[:, :1].to_numpy(), df.iloc[:, 1:].to_numpy(), test_size=0.15)[1].sum(axis=0)
    train_df = pd.concat([pd.DataFrame(X_train, columns=["image_id", "mask_image_id"]), pd.DataFrame(y_train, columns=disease_names)], axis=1)
    val_df = pd.concat([pd.DataFrame(X_val, columns=["image_id", "mask_image_id"]), pd.DataFrame(y_val, columns=disease_names)], axis=1)
    train_df.to_csv("train_73.csv")
    val_df.to_csv("val_13.csv")

    print("")


data_type = {"train": "train", "test": "test"}
# preprocess_classification_records(data_type["test"])
# preprocess_annotation_records(data_type["test"])
# dicom_to_pngs(data_type["test"])
dicom_to_pngs_unchanged(data_type["test"])
# make_csv_for_model(data_type["test"])
# train_test_split()