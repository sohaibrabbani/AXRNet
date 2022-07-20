import cv2
import numpy as np
import os
import pandas as pd
from configparser import ConfigParser

from PIL import Image

from generator import AugmentedImageSequence
from models.keras import ModelFactory
from keras import backend as kb
import mlflow


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def grad_cam(input_model, image, cls, layer_name, shape):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = kb.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = kb.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function(image)
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, shape, cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def create_cam(df_g, output_dir, image_source_dir, model, generator, class_names, df_images):
    """
    Create a CAM overlay image for the input image

    :param df_images:
    :param df_g: pandas.DataFrame, bboxes on the same image
    :param output_dir: str
    :param image_source_dir: str
    :param model: keras model
    :param generator: generator.AugmentedImageSequence
    :param class_names: list of str
    """
    file_name = df_g["image_id"]
    print(f"process image: {file_name}")

    # draw bbox with labels
    img_ori = cv2.imread(filename=os.path.join(image_source_dir + "_unchanged", file_name))
    gt_image_info = df_images.loc[df_images["file_name"] == file_name.split(".png")[0]]
    for index in [i for i, d in enumerate(df_g.iloc[3:].tolist()) if d == 1]:
        # label = df_g["label"]
        label = class_names[index]
        if label == "Infiltrate":
            label = "Infiltration"
        # index = class_names.index(label)

        output_path = os.path.join(output_dir, f'{label}.{file_name}')
        folder_path = os.path.join(output_dir, f'{label}')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        output_path_mask = os.path.join(folder_path, f'{label}.{file_name}')

        img_transformed = generator.load_image(file_name)

        # CAM overlay
        # Get the 512 input weights to the softmax.
        cam = grad_cam(model, [np.array([img_transformed])], index, "bn", (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.5)] = 0
        # heatmap[700:, :300] = 0
        mask_img = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(output_path_mask, np.where(mask_img > 0, 255, 0))
        # cv2.imwrite(output_path_mask, heatmap)
        img = img_ori
        # add label & rectangle
        # ratio = output dimension / 1024

        ratio = 1
        for _, row in gt_image_info.loc[gt_image_info["label"] == label].iterrows():
            x1 = int(row["x_min"] * ratio)
            y1 = int(row["y_min"] * ratio)
            x2 = int((row["x_max"]) * ratio)
            y2 = int((row["y_max"]) * ratio)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 15)
            cv2.putText(img, text=label, org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=(0, 0, 255), thickness=1)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        img = heatmap * 0.5 + img
        cv2.imwrite(output_path, img)


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    run_id = mlflow.list_run_infos("3")[0].run_id

    # default config
    output_dir_root = cp["DEFAULT"].get("output_dir")
    output_dir = output_dir_root + run_id
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    mask_image_source_dir = cp["DEFAULT"].get("mask_image_source_dir")
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir_root, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")
    original_weights_path = os.path.join(output_dir_root, f"original_{output_weights_name}")

    # CAM config
    bbox_list_file = cp["CAM"].get("bbox_list_file")
    use_best_weights = cp["CAM"].getboolean("use_best_weights")
    use_original_weights = cp["TEST"].getboolean("use_original_weights")

    print("** load model **")
    if use_original_weights:
        print("** use original weights **")
        model_weights_path = original_weights_path
    elif use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path,
        for_test=True)

    print("read bbox list file")
    df_images = pd.read_csv(bbox_list_file, header=None, skiprows=1)
    df_images = df_images.iloc[:, 1:]
    df_images.columns = ["file_name", "label", "x_min", "y_min", "x_max", "y_max"]

    df_test = pd.read_csv(os.path.join(output_dir, "test.csv"))

    print("create a generator for loading transformed images")
    cam_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(output_dir, "test.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=1,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=1,
        shuffle_on_epoch_end=False,
        status="test"
    )

    image_output_dir = os.path.join(output_dir, "cam")
    if not os.path.isdir(image_output_dir):
        os.makedirs(image_output_dir)

    print("create CAM")
    df_test.apply(
        lambda g: create_cam(
            df_g=g,
            output_dir=image_output_dir,
            image_source_dir=image_source_dir,
            model=model,
            generator=cam_sequence,
            class_names=class_names,
            df_images=df_images,
        ),
        axis=1,
    )


if __name__ == "__main__":
    # utils.filter_test_data_from_dataset()
    main()
