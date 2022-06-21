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


def dicom_to_pngs(output_dir):
    path = "./data/vindr-cxr/annotations/annotations_train.csv"
    image_root_path = "./data/vindr-cxr/train/"
    files = pd.read_csv(path)
    for i in files["image_id"]:
        image_name = i + ".dicom"
        ds = dcm.dcmread(os.path.join(image_root_path, image_name))
        img = ds.pixel_array
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            img = np.amax(img) - img
        img = img - np.min(img)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_NEAREST)
        img = ((np.maximum(img, 0)/img.max())*255).astype(np.uint8)
        cv2.imwrite(os.path.join("./data/vindr-cxr/train_pngs", image_name.replace(".dicom", ".png")), img)


def preprocess_classification_records():
    df = pd.read_csv("./data/vindr-cxr/annotations/image_labels_train.csv")
    # Filtered 5 diseases for our use
    df = df[["image_id", "rad_id", "Consolidation", "Nodule/Mass", "Pleural thickening", "Infiltration", "Pulmonary fibrosis"]]
    # Filtered patients with any disease (remove healthy patients)
    df = df[df.isin([1]).any(axis=1)]
    # Added all the classification of different radiologists
    df = df.groupby("image_id").sum()
    # Replaced greater values with 1
    df = df.where(df == 0, other=1)
    df.to_csv("./data/vindr-cxr/annotations/image_labels_train_5.csv")

def preprocess_annotation_records():
    df = pd.read_csv("./data/vindr-cxr/annotations/annotations_train.csv")
    disease_names = ["image_id", "rad_id", "Consolidation", "Nodule/Mass", "Pleural thickening", "Infiltration", "Pulmonary fibrosis"]
    df = df.loc[df['class_name'].isin(disease_names)]
    df.to_csv("./data/vindr-cxr/annotations/annotations_train_5.csv")


def make_annotation_masks_from_bboxes():
    df = pd.read_csv("./data/vindr-cxr/annotations/annotations_train.csv")




