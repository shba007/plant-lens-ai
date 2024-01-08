import os
import random

from PIL import Image
import deeplake


ROOT_PATH = "D:/Programming/Projects/Public/plant-lens/ai"
ANNOTATED_DATA_PATH = f"{ROOT_PATH}/data/annotated"
DATASET_DATA_PATH = f"{ROOT_PATH}/data/dataset"
TEMP_DATA_PATH = f"{ROOT_PATH}/data/temp"
DIMENSIONS = (224, 224)


# Find min and max items in all labels and also the labels names
def get_data_info(data):
    total = 0
    label_min = {"name": "", "count": 99999999}
    label_max = {"name": "", "count": -1}
    for label in data:
        if (len(data[label]) < label_min["count"]):
            label_min["name"] = label
            label_min["count"] = len(data[label])
        elif (len(data[label]) > label_max["count"]):
            label_max["name"] = label
            label_max["count"] = len(data[label])

        total += len(data[label])
        print({label: len(data[label])})

    print("\nTotal", total, "\nMin Label",
          label_min, "\nMax Label", label_max)


def display_image(path_or_image: str | Image.Image, title: str | None = None):
    image = Image.open(path_or_image) if isinstance(
        path_or_image, str) else path_or_image
    image.show(title=title)


def save_image(image: Image.Image, name: str):
    image.save(f"{TEMP_DATA_PATH}/{name}")
    return f"{TEMP_DATA_PATH}/{name}"


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


def main():
    data = {}

    # Load all filepath into a labels dict
    for root, dirs, files in os.walk(ANNOTATED_DATA_PATH):
        for file in files:
            full_path = os.path.normpath(root)
            base_path = os.path.normpath(ANNOTATED_DATA_PATH)
            label = full_path.split(base_path)[-1][1:]

            if label in data:
                data[label].append(file)
            else:
                data[label] = [file]

    data_classes = list(data.keys())

    get_data_info(data)

    training_dataset = {}
    testing_dataset = {}
    split_ratio = 0.8

    for label in data:
        items = data[label][:]
        split_index = int(len(items) * split_ratio)

        random.shuffle(items)

        training_dataset[label] = items[:split_index]
        testing_dataset[label] = items[split_index:]

    print("\nTraining Dataset")
    get_data_info(training_dataset)
    print("\nTesting Dataset")
    get_data_info(testing_dataset)

    for dataset in ['training', 'testing']:
        # Create the dataset locally
        ds = deeplake.empty(f'{DATASET_DATA_PATH}/{dataset}', overwrite=True)
        data = training_dataset if dataset == 'training' else testing_dataset

        with ds:
            ds.create_tensor('images', htype='image',
                             sample_compression='jpeg')
            ds.create_tensor('labels', htype='class_label', dtype='uint16',
                             class_names=data_classes)

            # ds.info.update(description='My first Deep Lake dataset')
            # ds.images.info.update(camera_type='SLR')

            for label in data:
                label_num = data_classes.index(label)
                for file in data[label]:
                    file_path = f"{ANNOTATED_DATA_PATH}/{label}/{file}"
                    padded_image = pad(file_path, 1)
                    resized_image = resize(padded_image, DIMENSIONS)
                    channel_resized_image = channel_resize(resized_image)

                    ds.append({'images': deeplake.read(save_image(
                        channel_resized_image, f"{label}-{file}")), 'labels': label_num})

            # commit_id = ds.commit('Added image of a cat')
            # print('Dataset in commit {} has {} samples'.format(commit_id, len(ds)))

            ds.summary()
            # ds.visualize()


if __name__ == "__main__":
    main()
