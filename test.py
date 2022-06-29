import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score
from utility import get_sample_counts
import mlflow
import shutil


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    mlflow.set_experiment("CheXNet Inference Experiments")
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id
    # run_id = mlflow.list_run_infos("0")[0].run_id
    # default config
    output_dir_root = cp["DEFAULT"].get("output_dir")
    output_dir = output_dir_root + run_id
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    # class_names = [x.lower() for x in class_names]
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    mask_image_source_dir = cp["DEFAULT"].get("mask_image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")
    use_original_weights = cp["TEST"].getboolean("use_original_weights")
    dataset_csv_dir = cp["TRAIN"].get("dataset_csv_dir")
    datasets = {"train": "train_73.csv", "val": "val_13.csv", "test": "test.csv"}

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")
    original_weights_path = os.path.join(output_dir_root, f"original_{output_weights_name}")

    # check output_dir, create it if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print(f"backup config file to {output_dir}")
    shutil.copy(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))

    for dataset_name in datasets:
        shutil.copy(os.path.join(dataset_csv_dir, datasets[dataset_name]), output_dir)

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, datasets["test"], class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError(f"""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print(f"** test_steps: {test_steps} **")

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

    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(output_dir, "test.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
        status="test"
    )

    print("** make prediction **")
    y_hat = model.predict_generator(test_sequence, verbose=1)
    y = test_sequence.get_y_true()

    test_log_path = os.path.join(output_dir, "test.log")
    print(f"** write log to {test_log_path} **")
    aurocs = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write(f"{class_names[i]}: {score}\n")
            mlflow.log_metric(class_names[i], score)
        mean_auroc = np.mean(aurocs)
        mlflow.log_metric("best_mean_auroc", mean_auroc)
        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auroc}\n")
        print(f"mean auroc: {mean_auroc}")
    mlflow.log_artifact(test_log_path)
    mlflow.log_artifact("./test.py")
    mlflow.end_run()

if __name__ == "__main__":
    main()