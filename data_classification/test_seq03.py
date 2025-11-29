import cv2
import pandas as pd
import os
import numpy as np
import joblib

#configs
IMAGE_FOLDER = "34759_final_project_rect/seq_03/image_02/data"
CSV_PATH = "/Users/franciscateixeira/Documents/DTU/1st semester/Perception for Autonomous Systems/project/DTU-34759-Final-Project/final_project_rectified_csv/seq03_detections_3d.csv"
MODEL_PATH = "modelo_final_otimizado.pkl"
OUTPUT_VIDEO = "seq3_classified.mp4"
FPS = 10

IMG_SIZE = (64, 64)

CLASSES = ['car', 'cyclist', 'pedestrian']

COLORS = {
    'car': (0, 0, 255),
    'pedestrian': (0, 255, 0),
    'cyclist': (255, 0, 0)
}

# ===================== FUNÇÕES ==========================

def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    flat = resized.flatten().reshape(1, -1)
    return flat


def main():
    print("load trained classifier")
    model = joblib.load(MODEL_PATH)


    print("load detection")
    df = pd.read_csv(CSV_PATH)

    images = [img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()

    if not images:
        print("error: no images found in", IMAGE_FOLDER)
        return

    first_img_path = os.path.join(IMAGE_FOLDER, images[0])
    frame0 = cv2.imread(first_img_path)
    h, w, c = frame0.shape

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

    print(f"processing {len(images)} frames...")

    # loop to create video
    for i, img_name in enumerate(images):

        frame_path = os.path.join(IMAGE_FOLDER, img_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        try:
            frame_id = int(img_name.split('.')[0])
        except ValueError:
            print(f"error extracting id from {img_name}")
            continue

        detections = df[df["frame"] == frame_id]


        for _, row in detections.iterrows():

            x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            X = preprocess_crop(crop)
            pred_idx = model.predict(X)[0]
            label = CLASSES[pred_idx]

            color = COLORS.get(label, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(img, f"frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(img)

    out.release()
    print("\ndone", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
