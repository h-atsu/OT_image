import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def pic_as_list(array):
    height, width = array.shape
    list_of_pixels = []
    for x in range(height):
        for y in range(width):
            color = array[x, y]
            list_of_pixels.append([x, y, color])
    return sorted(list_of_pixels, key=lambda x: (x[2], x[0], x[1]))


def transport_colors(start_array, target_array):
    height, width = start_array.shape
    list_start = pic_as_list(start_array)
    hist_target = [(target_array == color).sum() for color in range(256)]

    ind = 0
    for color in range(256):
        while hist_target[color] > 0:
            list_start[ind][2] = color
            hist_target[color] = hist_target[color] - 1
            ind = ind + 1

    target = np.zeros((height, width))
    for pixel in list_start:
        x, y, color = pixel
        target[x, y] = color
    return target


def transform(in_array, ref_array):
    """
    ndarray, ndarray -> PIL.image
    """
    height = in_array.shape[0]
    width = in_array.shape[1]
    final_picture = np.zeros((height, width, 3))
    for i in range(3):
        final_picture[:, :, i] = transport_colors(
            in_array[:, :, i], ref_array[:, :, i])
    final_picture = final_picture.astype(np.uint8)
    img = Image.fromarray(final_picture)
    return img


if __name__ == '__main__':
    result_string = 'data/out.png'
    start_string = 'data/input.png'
    target_string = 'data/ref.png'

    start_img = Image.open(start_string)
    target_img = Image.open(target_string)

    start_array = np.array(start_img)
    target_array = np.array(target_img)

    final_array = transform(start_array, target_array)
    plt.imshow(final_array)
    plt.show()
