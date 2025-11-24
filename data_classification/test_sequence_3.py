import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

# configs
IMG_SIZE = (64, 64)
CLASSES = ['car', 'cyclist', 'pedestrian']
MODEL_PATH = 'modelo_final_otimizado.pkl'


SEQ3_IMAGE_PATH = "34759_final_project_rect/seq_03/image_02/data/0000000000.png"


def sliding_window(image, step_size, window_size):
    """
    crops a sliding window across the image
    """
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


print("model loading")
model = joblib.load(MODEL_PATH)

image = cv2.imread(SEQ3_IMAGE_PATH)
if image is None:
    print(f"image not available in {SEQ3_IMAGE_PATH}")
    exit()

output_image = image.copy()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


print("analysing image")

windows_to_check = [
    (64, 64),  # try smaller windows
    (128, 128)  # try larger windows
]

for (winW, winH) in windows_to_check:
    for (x, y, crop) in sliding_window(gray_image, step_size=12, window_size=(winW, winH)):

        if y < 180:
            continue #ignore upper part of the image because of trees, for example

        # Se o recorte não tiver o tamanho certo, ignora
        if crop.shape[0] != winH or crop.shape[1] != winW:
            continue

        # resize to 64x64
        crop_resized = cv2.resize(crop, IMG_SIZE)

        crop_flat = crop_resized.flatten().reshape(1, -1)

        # predict probabilities
        probabilities = model.predict_proba(crop_flat)[0]
        best_class_idx = np.argmax(probabilities)
        confidence = probabilities[best_class_idx]

        # only draws if confidence is high
        if confidence > 0.99:
            label = CLASSES[best_class_idx]

            # colors
            if label == 'car':
                color = (255, 0, 0)
            elif label == 'cyclist':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # draw
            cv2.rectangle(output_image, (x, y), (x + winW, y + winH), color, 2)
            cv2.putText(output_image, f"{label}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Classificação na Sequência 3 (Janela Deslizante)")
plt.axis('off')
plt.show()

cv2.imwrite("resultado_seq3.png", output_image)
print("Resultado guardado como 'resultado_seq3.png'")