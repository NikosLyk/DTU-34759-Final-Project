import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# configurations
IMG_SIZE = (64, 64)
CLASSES = ['car', 'cyclist', 'pedestrian']
BASE_TRAIN_PATH = "train/augmentation"
BASE_VAL_PATH = "val"


def load_images_from_folder(folder, label_idx):
    data = []
    labels = []

    for filename in os.listdir(folder):

        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, IMG_SIZE)
            data.append(resized.flatten())
            labels.append(label_idx)

    return data, labels


def load_dataset(base_path):
    X = []
    y = []
    print(f"Loading dataset from: {base_path}")

    for class_name in CLASSES:
        idx = CLASSES.index(class_name)
        folder_path = os.path.join(base_path, class_name)

        images, targets = load_images_from_folder(folder_path, idx)
        X.extend(images)
        y.extend(targets)
        print(f"  - {class_name}: {len(images)} loaded images")

    return np.array(X), np.array(y)



print("preparing to train the model")
X_train, y_train = load_dataset(BASE_TRAIN_PATH)
X_val, y_val = load_dataset(BASE_VAL_PATH)

if len(X_train) == 0:
    print("No training images were found")
    exit()

# Se não tiveres validação ainda, usa uma parte do treino para testar
if len(X_val) == 0:
    print("No validation images were found, splitting training data...")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#grid search
print("\n otimizing hyperparameters with Grid Search")

# define base Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.90)),
    ('svm', SVC(kernel='rbf', probability=True))
])

# parameter to test
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.001],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print(f"\nbest parameters founf: {grid.best_params_}")
print(f"best accuracy in training: {grid.best_score_*100:.2f}%")

best_model = grid.best_estimator_

# final evaluation
print("\nvalidation results with the best model:")
predictions = best_model.predict(X_val)
print(f"accuracy: {accuracy_score(y_val, predictions) * 100:.2f}%")
print(classification_report(y_val, predictions, target_names=CLASSES))

# confusion matrix
cm = confusion_matrix(y_val, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title("confusion matrix")
plt.show()

# save
joblib.dump(best_model, 'modelo_final_otimizado.pkl')
print("saved")