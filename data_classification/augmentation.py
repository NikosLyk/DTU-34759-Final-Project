import cv2
import os
import imutils
import numpy as np

pedestrian_folder = "train/pedestrian"
car_folder = "train/car"
cyclist_folder = "train/cyclist"

output_pedestrian_folder = "train/augmentation/pedestrian"
output_car_folder = "train/augmentation/car"
output_cyclist_folder = "train/augmentation/cyclist"

os.makedirs(output_pedestrian_folder, exist_ok=True)
os.makedirs(output_car_folder, exist_ok=True)
os.makedirs(output_cyclist_folder, exist_ok=True)


def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss * 50
    return np.clip(noisy, 0, 255).astype('uint8')


def augment_images(input_folder, output_folder):

    for file in os.listdir(input_folder):
        path = os.path.join(input_folder, file)
        img = cv2.imread(path)

        if img is None:
            print("Could not read", file)
            continue

        # original
        cv2.imwrite(output_folder + "/" + file, img)
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
        blur = cv2.GaussianBlur(img, (9,9), 0)
        cv2.imwrite(output_folder + "/blur_" + file, blur)

        # brightness +40
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=40)
        cv2.imwrite(output_folder + "/bright_" + file, bright)

        # 7. Low Contrast
        low_cont = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
        cv2.imwrite(output_folder + "/low_contrast_" + file, low_cont)

        # 8. Noisy
        noisy = add_noise(img)
        cv2.imwrite(output_folder + "/noisy_" + file, noisy)

        small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_folder + "/pixelated_" + file, pixelated)

augment_images(pedestrian_folder, output_pedestrian_folder)
augment_images(car_folder, output_car_folder)
augment_images(cyclist_folder, output_cyclist_folder)
