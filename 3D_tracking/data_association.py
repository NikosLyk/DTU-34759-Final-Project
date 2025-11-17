import numpy as np
from scipy.optimize import linear_sum_assignment
import IoU


def assosiate_detections_to_kalmans(kalman_predictions, detections, depth_threshold, iou_threshold=0.3):

    if len(kalman_predictions) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(kalman_predictions)), np.arange(len(detections))

    num_kalmans = len(kalman_predictions)
    num_detections = len(detections)

    cost_matrix = np.ones((num_kalmans, num_detections)) * 1e9  # Initialize with very high cost, for invalid matches

    for k, kalman in enumerate(kalman_predictions):
        for d, detection in enumerate(detections):
            iou = IoU.calculate_iou(kalman.get_box_corners(), detection.box)  # TODO: not sure what the detections is going to look like
            depth_diff = abs(kalman.x[2] - detection.depth) # TODO: aslo need to figure the depth of the detections out

            # IoU is used for cost calculation, while the depth is only used for gating. TODO: Check if this is sufficient
            if iou > iou_threshold and depth_diff < depth_threshold:
                cost_matrix[k, d] = 1.0 - iou
            # If the gating check fails, the cost remains 1e9

    # Running the hungarian algorithm
    kalman_indices, detection_indices = linear_sum_assignment(cost_matrix)

    matches = []

    unmatched_kalmans = list(range(num_kalmans))
    unmatched_detections = list(range(num_detections))

    for k_idx, d_idx in zip(kalman_indices, detection_indices):
        if cost_matrix[k_idx, d_idx] < 1e8:
            matches.append([k_idx, d_idx])

            if k_idx in unmatched_kalmans:
                unmatched_kalmans.remove(k_idx)
            if d_idx in unmatched_detections:
                unmatched_detections.remove(d_idx)

    return np.array(matches), np.array(unmatched_kalmans), np.array(unmatched_detections)