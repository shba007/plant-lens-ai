import os
import json
import math
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models

DATASET_TYPE = "plant"
ROOT_PATH = "D:/Programming/Projects/Public/plant-lens/ai"
DEVELOPMENT_MODEL_PATH = f"{ROOT_PATH}/model/develop/{DATASET_TYPE}"
ANNOTATED_DATA_PATH = f"{ROOT_PATH}/data/annotated/{DATASET_TYPE}"
DIMENSIONS = (224, 224)


# function which takes a image + aspect ratio output padded image
def pad(path: str, target_aspect_ratio: float, fill_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    image = Image.open(path)
    aspect_ratio = image.width / image.height

    if aspect_ratio == target_aspect_ratio:
        return image

    if aspect_ratio < target_aspect_ratio:
        width = int(image.height * target_aspect_ratio)
        height = image.height
    else:
        width = image.width
        height = int(image.width / target_aspect_ratio)

    padded_image = Image.new(image.mode, (width, height), fill_color)
    paste_position = ((width - image.width) // 2,
                      (height - image.height) // 2)

    padded_image.paste(image, paste_position)
    return padded_image


# a function which takes an image resizes it with the current aspect ratio and the given width
def resize(image: Image.Image, dimensions: tuple[int, int]) -> Image.Image:
    return image.resize(dimensions)


def channel_resize(image: Image.Image) -> Image.Image:
    num_channels = len(image.getbands())
    return image.convert('RGB') if num_channels == 1 else image


# Define a preprocessing function
def preprocess_image(image) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.math.divide(image, 255.0)

    return image


def save_features(features, labels):
    grouped_features = {}

    for label, feature in zip(labels, features):
        if label in grouped_features:
            grouped_features[int(label)].append(feature)
        else:
            grouped_features[int(label)] = [feature]

    with open("features.json", 'w') as json_file:
        json.dump(grouped_features, json_file)

    print(f"Averaged Tensor has been dumped to {'features.json'}.")


def load_model(VERSION_TAG):
    return models.load_model(f'{DEVELOPMENT_MODEL_PATH}/v{VERSION_TAG}.h5')


def main():
    data: dict[str, list[list[int]]] = {}

    # Load all filepath into a labels dict
    ANNOTATED_DATA_IMAGE_PATH = f"{ANNOTATED_DATA_PATH}/images"
    for root, dirs, files in os.walk(ANNOTATED_DATA_IMAGE_PATH):
        for file in files:
            full_path = os.path.normpath(root)
            base_path = os.path.normpath(ANNOTATED_DATA_IMAGE_PATH)
            label = full_path.split(base_path)[-1][1:]

            if label in data:
                data[label].append(file)
            else:
                data[label] = [file]

    data_classes = list(data.keys())
    print(data_classes)
    feature_extractor_model = load_model("0.6.7-50")
    feature_extractor_model.summary()

    images: list[tf.Tensor] = []
    labels: list[str] = []

    for label in data:
        for index, file in enumerate(data[label][:min(len(data[label]), 32)]):
            file_path = f"{ANNOTATED_DATA_PATH}/images/{label}/{file}"
            padded_image = pad(file_path, 1)
            resized_image = resize(padded_image, DIMENSIONS)
            channel_resized_image = channel_resize(resized_image)

            labels.append(data_classes.index(label))
            images.append(preprocess_image(channel_resized_image))

    labels = tf.cast(labels, tf.int32).numpy()
    features = []

    bucket_size = 2048
    for i in range(0, len(images), bucket_size):
        start = i
        end = min(i + bucket_size, len(images))
        slicedImages = images[start:end]
        # Convert the sliced images to a TensorFlow tensor
        slicedImagesTensor = tf.convert_to_tensor(slicedImages)
        # Perform batch prediction
        slicedFeatures = feature_extractor_model.predict(slicedImagesTensor)
        # Append the batch of features to the list
        features.extend(slicedFeatures.tolist())
        print(f"Batch {i//bucket_size+1}: Processed {end} images")

    # Convert the list of features to a NumPy array
    # features = tf.concat(features, axis=0).numpy()

    save_features(features, labels)


if __name__ == "__main__":
    main()
