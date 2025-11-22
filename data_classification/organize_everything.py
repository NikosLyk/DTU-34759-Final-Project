import cv2
import os

BASE = "/Users/franciscateixeira/Documents/DTU/1st semester/Perception for Autonomous Systems/DTU-34759-Final-Project/data_classification/34759_final_project_rect"

def extract_crops(seq_name, output_path):
    seq_path = BASE + "/" + seq_name
    labels_file = seq_path +  "/labels.txt"
    images_path = seq_path + "/image_02/data"

    if not os.path.exists(labels_file):
        print(f"No labels.txt in {seq_name}")
        return

    print(f"Extracting crops from {seq_name}")

    with open(labels_file, "r") as f:
        for i, line in enumerate(f):
            fields = line.strip().split()
            frame = int(fields[0])
            obj_type = fields[2]   # object type in the image

            # bounding box
            left = int(float(fields[6]))
            up = int(float(fields[7]))
            right = int(float(fields[8]))
            down = int(float(fields[9]))

            if seq_name == "seq_01":
                img_file = os.path.join(images_path, f"{frame:06d}.png")

            elif seq_name == "seq_02":
                img_file = os.path.join(images_path, f"{frame:010d}.png")

            img = cv2.imread(img_file)

            if img is None:
                print(f"Error reading {img_file}")
                continue

            crop = img[up:down, left:right]

            if crop.size == 0:
                print(f" Empty crop on {img_file}")
                continue

            crop = cv2.resize(crop, (128, 128))

            # save crop
            save_dir = os.path.join(output_path, obj_type.lower())
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{seq_name}_crop_{i}.png")
            cv2.imwrite(save_path, crop)


if __name__ == "__main__":
    # extract sequence crops
    extract_crops("seq_01", BASE + "/val")
    extract_crops("seq_02", BASE + "/val")

    print("\nAll extracted")
