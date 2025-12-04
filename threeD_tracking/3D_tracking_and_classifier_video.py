import numpy as np
import cv2
import threeD_tracker
import pandas as pd
from pathlib import Path
import joblib

SEQ_ID = "seq_03"  # <--- Change this one variable to update the whole system
ROOT   = Path.cwd().parent

CLASSIFIER_PATH = "/Users/franciscateixeira/Documents/DTU/1st semester/Perception for Autonomous Systems/project/DTU-34759-Final-Project/data_classification/modelo_final_otimizado.pkl"   # ADDED
classifier = joblib.load(CLASSIFIER_PATH)               # ADDED
IMG_SIZE = (64, 64)                                     # ADDED
CLASSES = ['car', 'cyclist', 'pedestrian']              # ADDED

rect_root = ROOT / "data_classification" / "34759_final_project_rect" / SEQ_ID
csv_filename = f"{SEQ_ID.replace('_', '')}_detections_3d.csv"

dirs = {
    "rect":  rect_root / "image_02" / "data",
    "csv":   ROOT / "rectified_detection_csv" / csv_filename,
    "video": ROOT / "output" / "tracking_videos"
}

# Use dirs['rect'], dirs['csv'], etc.
frames = sorted(dirs["rect"].glob("*.png"))

def prepare_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    # Clamp coordinates to image boundaries
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))

    # If invalid box (can happen with Kalman), skip classification
    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # resize with padding — same logic as training
    h, w = gray.shape
    target_w, target_h = IMG_SIZE
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w <= 0 or new_h <= 0:
        return None

    resized = cv2.resize(gray, (new_w, new_h))

    padded = np.zeros((target_h, target_w), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return padded.flatten()


def classify_bbox(img, x1, y1, x2, y2):
    feat = prepare_crop(img, x1, y1, x2, y2)
    if feat is None:
        return "unknown"   # fallback
    pred = classifier.predict(feat.reshape(1, -1))[0]
    return CLASSES[pred]

# Visualization helper function
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

def draw_tracks(frame, tracker):
    img = frame.copy()

    # --- 1. Draw ALL Trajectories (Background) ---
    # We draw these first so they appear "behind" the bounding boxes
    for obj_id, history in tracker.trajectories.items():
        # Get color based on ID
        color = [int(c) for c in COLORS[obj_id % len(COLORS)]]

        # Extract the last 30 points
        pts = []
        for h in history[-30:]:
            # Ensure center is a simple list of ints
            c_x = int(float(h['center'][0]))
            c_y = int(float(h['center'][1]))
            pts.append((c_x, c_y))

        if len(pts) > 1:
            # Draw connected lines
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], color, 2)

    # --- 2. Draw Active Boxes (Foreground) ---
    # Only draw boxes for objects matched IN THIS FRAME
    for kalman in tracker.active_kalmans:

        # THE FIX: Skip drawing boxes for objects currently "coasting" (lost)
        if kalman.missed_frames > 1000:
            continue

        obj_id = kalman.id
        color = [int(c) for c in COLORS[obj_id % len(COLORS)]]

        # Safe Box Extraction (using the Flatten fix from before)
        raw_box = kalman.get_box_corners()
        flat_box = np.array(raw_box).flatten()
        x1, y1, x2, y2 = [int(float(x)) for x in flat_box[:4]]

        # Safe Depth Extraction
        depth = float(np.array(kalman.x).flatten()[2])

        # Draw Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cls = getattr(kalman, "cls", "unknown")
        # Draw Text Label
        label = f"{cls} | ID:{obj_id} Z:{depth:.2f}m"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Text Background
        cv2.rectangle(img, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
        # Text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    total_trackers = len(tracker.active_kalmans)
    new_detections = tracker.n_detections
    status_text = f"Trackers: {total_trackers} (Detections: {new_detections})"

    # Position: Top-left corner
    text_pos = (20, 40)

    # Draw a black border for the text (so it's readable on light backgrounds)
    cv2.putText(img, status_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 4) # Thickness 4 (Outline)

    # Draw white text on top
    cv2.putText(img, status_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2) # Thickness 2 (Body)

    return img


df = pd.read_csv(dirs["csv"])

# Groups the detections by frames, needed for use in kalman thing
grouped_frames = df.groupby("frame")

first_frame = cv2.imread(str(frames[0]))
height, width, _ = first_frame.shape
frame_size = (width, height)

output_video_path = dirs["video"] / f"{SEQ_ID}_tracked.mp4"
fps = 10

out = cv2.VideoWriter(
    str(output_video_path),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    frame_size
)

# Create tracker
# Using delta time of 1
tracker = threeD_tracker.Tracker3d(1, depth_threshold=15, max_missed_frames=35, max_pixel_dist=250)

for frame_number, detections_in_frame in grouped_frames:
    #frame_number += 1 # TODO: Double check this
    detections_in_frame['width'] = detections_in_frame['x2'] - detections_in_frame['x1']
    detections_in_frame['height'] = detections_in_frame['y2'] - detections_in_frame['y1']
    detections_in_frame['x_pos'] = detections_in_frame['x1'] + detections_in_frame['width'] / 2
    detections_in_frame['y_pos'] = detections_in_frame['y1'] + detections_in_frame['height'] / 2

    detections_in_frame = detections_in_frame[detections_in_frame['confidence'] > 0.35] # 0.45 works well

    detections = detections_in_frame[['x_pos', 'y_pos', 'depth_m', 'width', 'height', 'x1', 'y1', 'x2', 'y2']].values

    tracker.update_tracker(detections)
    #print(f"frame count: {tracker.frame_count}")
    #print(f"detections: {len(detections)}")

    # Load rectified image + depth map
    img = cv2.imread(str(frames[frame_number]))

    for kalman in tracker.active_kalmans:
        raw = kalman.get_box_corners()
        if raw is None:
            continue

        flat = np.array(raw).flatten()
        x1, y1, x2, y2 = [int(float(x)) for x in flat[:4]]

        cls = classify_bbox(img, x1, y1, x2, y2)
        kalman.cls = cls  # attach label to tracker label

    vis_frame = draw_tracks(img, tracker)
    out.write(vis_frame)

    if frame_number % 50 == 0:
        print(f"Processed frame {frame_number}...")

out.release()
print("Video generation complete.")
