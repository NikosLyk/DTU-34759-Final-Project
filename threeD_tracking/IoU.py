import numpy as np


def calculate_iou(boxA, boxB):
    """
    Calculates the intersection over uniouon of the bounding boxes of two detected objects

    Parameters:
    ----------
    boxA : list or tuple
        A list of 4 numbers representing the coordinates of the first
        bounding box in the format [x1, y1, x2, y2].
        (x1, y1) = top-left corner
        (x2, y2) = bottom-right corner

    boxB : list or tuple
        A list of 4 numbers representing the coordinates of the second
        bounding box in the format [x1, y1, x2, y2].

    Returns:
    -------
    float
        The IoU score, a value between 0.0 and 1.0.
    """
    inter_x1 = max(boxA[0], boxB[0])
    inter_y1 = max(boxA[1], boxB[1])
    inter_x2 = min(boxA[2], boxB[2])
    inter_y2 = min(boxA[3], boxB[3])

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    inter_area = inter_width * inter_height

    boxA_width = boxA[2] - boxA[0]
    boxA_height = boxA[3] - boxA[1]
    boxA_area = boxA_width * boxA_height

    boxB_width = boxB[2] - boxB[0]
    boxB_height = boxB[3] - boxB[1]
    boxB_area = boxB_width * boxB_height

    union_area = boxA_area + boxB_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area

    return iou