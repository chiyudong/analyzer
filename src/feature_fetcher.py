import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
import utility

# Set your dataset folder, in this folder, it may contain more than one datasets.
# e.g.,
#  tensorflow_datasets/
#               |_____nyu_rot_dataset_converted_externally_to_rlds/
#                                                       |_________0.0.1/
#                                                                   |_____dataset_info.json
#
# Note: In windows, the folder dir looks like : "C:\\Users\\john\\tensorflow_datasets\\"
DATASET_DIR = ""

# Set a feature, normally, we investigate "image"
INITIAL_FEATURE = "image"  # highres_image, input_point_cloud


# Select a dataset, could be iterate in the main function.
DATASET_IDX = 1


def fetch_features(dataset_dir, initial_features, dataset_idx):
    """
    Access the dataset and fetch proper fetchers, and save to the ../example folder
    """
    dataset = utility.DATASETS_SORTED[dataset_idx]
    dataset_folder = utility.get_dataset_folder(dataset, dataset_dir)

    if not dataset_folder:
        raise ValueError(f"{dataset} not exists.")

    # Build a database.
    db = tfds.builder_from_directory(builder_dir=dataset_folder)

    display_key = utility.get_feature_key(
        initial_key=initial_features, features=db.info.features["steps"]["observation"]
    )

    ds = db.as_dataset(split="train[:10]")  # take only first 10 episodes
    episode = next(iter(ds), 2)

    if display_key == "image" or "highres_image":
        images = [step["observation"][display_key] for step in episode["steps"]]
        images = [Image.fromarray(image.numpy()) for image in images]

        # Save the git into ./example
        path_dir = "".join(["../example/", str(dataset_idx), "_", dataset, ".gif"])
        utility.save_as_gif(images, path=path_dir)

    elif display_key == "input_point_cloud":
        point_clouds = [
            step["observation"]["input_point_cloud"] for step in episode["steps"]
        ]
        pcnumpy = np.array(point_clouds)
        # Save the cloud to ./example
        _ = utility.numpyarray_to_open3d(pcnumpy)

    # other elements of the episode step --> this may vary for each dataset
    for elem in next(iter(episode["steps"])).items():
        print(elem)


if __name__ == "__main__":

    if not DATASET_DIR:
        raise ValueError("DATASET_DIR not set.")

    """
        # Uncomment if you want to iteratively download and save
        for i in range(5, 7):
            fetch_features(
                dataset_dir=DATASET_DIR,
                initial_features=INITIAL_FEATURE,
                dataset_idx=i,
            )
    """
    fetch_features(
        dataset_dir=DATASET_DIR,
        initial_features=INITIAL_FEATURE,
        dataset_idx=DATASET_IDX,
    )
