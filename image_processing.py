import numpy as np
import cv2
import csv
import os
import argparse
from skimage import exposure


def random_brightness_adjust(image):
    y, u, v = cv2.split(image)
    y = np.asarray(y, dtype=np.int32)

    factor = int(np.random.normal(loc=0.0, scale=1.0, size=None) * 20)
    y += factor
    y = np.asarray(np.clip(y, 0, 255), dtype=np.uint8)
    return cv2.merge((y, u, v))


def rescale_intensity(img):
    pa, pb = np.percentile(img, (5, 95))
    img_rescale = exposure.rescale_intensity(img, in_range=(pa, pb))
    return img_rescale


def process_image(image):
    image = image[60:120, ]
    image = rescale_intensity(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image


def normalize_image(image):
    return (image / 127.5) - 1.


def preprocess_data(input_file, output_file, out_folder):
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        csv_in = list(reader)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print("Processing {0} * 3 images...".format(len(csv_in)))
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        for entry in csv_in:
            center_img = entry[0].strip()
            left_img = entry[1].strip()
            right_img = entry[2].strip()
            entry[0] = out_folder + "/" + os.path.basename(entry[0])
            entry[1] = out_folder + "/" + os.path.basename(entry[1])
            entry[2] = out_folder + "/" + os.path.basename(entry[2])
            cv2.imwrite(entry[0], process_image(cv2.imread(center_img, cv2.IMREAD_COLOR)))
            cv2.imwrite(entry[1], process_image(cv2.imread(left_img, cv2.IMREAD_COLOR)))
            cv2.imwrite(entry[2], process_image(cv2.imread(right_img, cv2.IMREAD_COLOR)))
            writer.writerow(entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image pre processing')
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    parser.add_argument('--img_dir')
    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_file, args.img_dir)