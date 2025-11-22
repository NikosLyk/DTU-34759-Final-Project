import cv2
import numpy as np
import os
import imutils

pedestrian_folder = "train/pedestrian"
car_folder = "train/car"
cyclist_folder = "train/cyclist"

output_pedestrian_folder = "train/augmentation/pedestrian"
output_car_folder = "train/augmentation/car"
output_cyclist_folder = "train/augmentation/cyclist"

os.makedirs(output_pedestrian_folder, exist_ok=True)
os.makedirs(output_car_folder, exist_ok=True)
os.makedirs(output_cyclist_folder, exist_ok=True)

def augment_images(input_folder, output_folder):

    for file in os.listdir(input_folder):
        path = os.path.join(input_folder, file)
        img = cv2.imread(path)

        if img is None:
            print("Could not read", file)
            continue

        h, w, d = img.shape

        # flip horizontal

        flip = cv2.flip(img, 1)
        cv2.imwrite(output_folder + "/flip_" +  file, flip)

        # rotate +15°
        center = (w // 2, h // 2)
        rot1 = imutils.rotate_bound(img, 15)
        cv2.imwrite(output_folder + "/rot15_" + file, rot1)

        # rotate -15°
        rot2 = imutils.rotate_bound(img, -15)
        cv2.imwrite(output_folder + "/rot-15_" + file, rot2)

        # gaussian blur
        blur = cv2.GaussianBlur(img, (5,5), 0)
        cv2.imwrite(output_folder + "/blur_" + file, blur)

        # brightness +40
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=40)
        cv2.imwrite(output_folder + "/bright_" + file, bright)

augment_images(pedestrian_folder, output_pedestrian_folder)
augment_images(car_folder, output_car_folder)
augment_images(cyclist_folder, output_cyclist_folder)
