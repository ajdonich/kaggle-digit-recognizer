import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mnist_image(pixel_data, label_data, img_rows, img_cols):

    data_row_index = 0
    m, data_cols = pixel_data.shape
    image = np.zeros([img_rows*28, img_cols*28], dtype=np.uint8)

    # Verify request does not exceed total amount of data
    assert (m * data_cols) >= (img_rows*28 * img_cols*28), "Not enough pixel_data"

    # Reshape pixels rows into an image matrix
    for row in range(0, img_rows*28, 28):
        for col in range(0, img_cols*28, 28):

            img_pixel_data = pixel_data.iloc[data_row_index, :].values.reshape(28, 28)
            image[row:(row+28), col:(col+28)] = img_pixel_data
            data_row_index += 1

    # Dump data labels to stdout for visual comparison
    labels = label_data.iloc[0:img_rows*img_cols].values.reshape(img_rows, img_cols)
    print("\n" + str(labels))

    # Display image in reversed grey scale
    plt.imshow(image, cmap='gray_r')
    plt.show()

    return True

def plot_error_image(pixel_data, label_data, pred_data, img_rows, img_cols):

    # Separate success predictions from plot data
    fail_indices = np.nonzero(label_data - pred_data)
    fail_labels_pred = pred_data.iloc[fail_indices[0]]
    fail_labels_truth = label_data.iloc[fail_indices[0]]
    fail_pixels = pixel_data.iloc[fail_indices[0], :]

    print("Ground Truth Labels: ")
    print(fail_labels_truth.iloc[0:img_rows*img_cols].values.reshape(img_rows, img_cols))

    return plot_mnist_image(fail_pixels, fail_labels_pred, img_rows, img_cols)


def plot_label_dist(label_data):

    sns.set(color_codes=True)
    sns.distplot(label_data, bins=10, kde=False, rug=True)
    plt.show()

    return True

def plot_train_hist(flat_history, conv_history):

    # List all data in history
    print(flat_history.history.keys())
    print(conv_history.history.keys())


    # Plot with respect to accuracy
    plt.figure(1)
    plt.plot(flat_history.history['acc'])
    plt.plot(flat_history.history['val_acc'])
    plt.title('Flat Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.figure(2)
    plt.plot(conv_history.history['acc'])
    plt.plot(conv_history.history['val_acc'])
    plt.title('Conv Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    # Plot with respect to loss
    plt.figure(3)
    plt.plot(flat_history.history['loss'])
    plt.plot(flat_history.history['val_loss'])
    plt.title('Flat Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.figure(4)
    plt.plot(conv_history.history['loss'])
    plt.plot(conv_history.history['val_loss'])
    plt.title('Flat Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()