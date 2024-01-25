import math
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# from lib.utils import convert_box

COLUM_SIZE = 8


def peek_labels(dataset, mode="preview", class_names=None):
    # Get one batch from the dataset
    batch = dataset.take(1)

    if mode == "preview":
        # bboxes
        images, labels = next(iter(batch))
    else:
        images, labels, predictions = next(iter(batch))

    col_size = COLUM_SIZE
    row_size = math.ceil(len(images)/col_size)

    fig, axs = plt.subplots(
        row_size, col_size, figsize=(col_size*2, row_size*2))

    for i in range(len(images)):
        if row_size > 1:
            col = i % col_size
            row = math.floor(i / col_size)
            index = (row, col)
        else:
            index = i

        axs[index].imshow(images[i])
        if mode != "preview":
            # Create a rectangle patch for the border
            border = Rectangle((0, 0), images[i].shape[0], images[i].shape[1], linewidth=5, edgecolor='g'
                               if int(labels[i]) == int(predictions[i]) else 'r', facecolor='none')
            # Add the border to the axis
            axs[index].add_patch(border)

        # Add the border to the axis
        display_label = class_names[labels[i]] if (
            class_names != None) else labels[i]
        axs[index].set_title(f"Label: {display_label}" if mode ==
                             "preview" else f"Label: {labels[i]}, Pred: {predictions[i]}")

        axs[index].set_xticks([])
        axs[index].set_yticks([])


def peek_pairs(dataset, input="dataset", mode="preview"):
    def is_same(label):
        return "Same" if label > 0.5 else "Diff"

    if input == "dataset":
        # Get one batch from the dataset
        batch = dataset.take(1)
        unpacked_batch = next(iter(batch))
    else:
        unpacked_batch = dataset

    if mode == "preview":
        x, y = unpacked_batch
        x1, x2 = tf.unstack(x, axis=0)
        image_pairs = tf.concat(
            [x1[:, tf.newaxis], x2[:, tf.newaxis]], axis=1).numpy()
        similarity = y.numpy()
    else:
        x, y, y_hat = unpacked_batch
        x1, x2 = tf.unstack(x, axis=0)
        image_pairs = tf.concat(
            [x1[:, tf.newaxis], x2[:, tf.newaxis]], axis=1).numpy()
        similarity = y.numpy()
        predictions = y_hat.numpy()
    # Split the input and target tensors and convert to numpy arrays

    col_size = int(COLUM_SIZE/2)
    row_size = math.ceil(len(image_pairs)/col_size)

    fig, axs = plt.subplots(
        row_size, col_size, figsize=(col_size*4, row_size*2))

    for i in range(len(image_pairs)):
        if row_size > 1:
            col = i % col_size
            row = math.floor(i / col_size)
            index = (row, col)
        else:
            index = i

        subplots = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=axs[index])

        for j in range(2):
            ax = plt.Subplot(fig, subplots[j])
            ax.imshow(image_pairs[i][j])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

            if mode != "preview":
                # Create a rectangle patch for the border
                border = Rectangle((0, 0), image_pairs[i][0].shape[0], image_pairs[i][0].shape[1], linewidth=5, edgecolor='g'
                                   if similarity[i] == int(predictions[i] > 0.5) else 'r', facecolor='none')
                # Add the border to the axis
                ax.add_patch(border)

        # Add the border to the axis
        axs[index].set_title(f"Label: {is_same(similarity[i])}" if mode ==
                             "preview" else f"Label: {is_same(similarity[i])}, Pred: {is_same(predictions[i])}")
        axs[index].axis("off")


def peek_triplet(dataset, mode="preview"):
    def is_same(label):
        return "Same" if label > 0.5 else "Diff"
    # Get one batch from the dataset
    batch = dataset.take(1)

    if mode == "preview":
        x, y = next(iter(batch))
        x1, x2, x3 = tf.unstack(x, axis=0)
        image_triplet = tf.concat(
            [x1[:, tf.newaxis], x2[:, tf.newaxis], x3[:, tf.newaxis]], axis=1).numpy()
    else:
        x, y, y_hat = next(iter(batch))
        x1, x2, x3 = tf.unstack(x, axis=0)
        image_triplet = tf.concat(
            [x1[:, tf.newaxis], x2[:, tf.newaxis], x3[:, tf.newaxis]], axis=1).numpy()
        similarity, dissimilarity = tf.unstack(y, axis=0)
        similarity_predictions, dissimilarity_predictions = tf.unstack(
            y_hat, axis=0)
    # Split the input and target tensors and convert to numpy arrays

    col_size = int(COLUM_SIZE/2)
    row_size = math.ceil(len(image_triplet)/col_size)

    fig, axs = plt.subplots(
        row_size, col_size, figsize=(col_size*4, row_size*2))

    for i in range(len(image_triplet)):
        if row_size > 1:
            col = i % col_size
            row = math.floor(i / col_size)
            index = (row, col)
        else:
            index = i

        subplots = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=axs[index])

        for j in range(3):
            ax = plt.Subplot(fig, subplots[j])
            ax.imshow(image_triplet[i][j])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

            # if mode != "preview":
            #     # Create a rectangle patch for the border
            #     border = Rectangle((0, 0), image_triplet[i][0].shape[0], image_triplet[i][0].shape[1], linewidth=5, edgecolor='g'
            #                        if similarity[i] == int(predictions[i] > 0.5) else 'r', facecolor='none')
            #     # Add the border to the axis
            #     ax.add_patch(border)

        # Add the border to the axis
        axs[index].set_title(f"Similarity: True, Dissimilarity: False" if mode ==
                             "preview" else f"Similarity: {similarity_predictions}, Dissimilarity: {dissimilarity_predictions}")
        axs[index].axis("off")


def peek_boxes(dataset, dimensions, mode="preview"):
    # Get one batch from the dataset
    batch = dataset.take(1)
    print(batch)

    for a in batch:
        print(a)
        print(a.shape)

    if mode == "preview":
        images, labels, bboxes = next(iter(batch))
    else:
        images, bboxes, predictions = next(iter(batch))

    col_size = int(COLUM_SIZE/2)
    row_size = math.ceil(len(images)/col_size)

    fig, axs = plt.subplots(
        row_size, col_size, figsize=(col_size*4, row_size*2))

    for i in range(len(images)):
        if row_size > 1:
            col = i % col_size
            row = math.floor(i / col_size)
            index = (row, col)
        else:
            index = i

        axs[index].imshow(images[i])

        def pipeline(bboxes, color='g'):
            for bbox in bboxes:
                # Convert CCWH to XYXY
                xmin, ymin, xmax, ymax = convert_box(
                    bbox, images[i].shape, init_format="CCWH", init_normalized=True, final_format="XYXY")
                # Create a rectangle patch for the border
                box = Rectangle((xmin, ymin), xmax, ymax,
                                linewidth=5, edgecolor=color, facecolor='none')
                # Add the border to the axis
                axs[index].add_patch(box)

        pipeline(bboxes)
        if mode != "preview":
            pipeline(predictions, 'r')

        # Add the border to the axis
        # axs[index].set_title(f"Label: {labels[i]}" if mode == "preview" else f"Label: {labels[i]}, Pred: {predictions[i]}")

        axs[index].set_xticks([])
        axs[index].set_yticks([])
