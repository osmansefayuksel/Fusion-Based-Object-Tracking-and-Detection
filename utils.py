import numpy as np
import cv2

def initialize_tracker(tracker_type):
    if tracker_type == 'boosting':
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'mil':
        return cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'kcf':
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'tld':
        return cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'medianflow':
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'mosse':
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "csrt":
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")


def compute_iou(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou



def compute_center_location_error(gt_bbox, pred_bbox):
   
    gt_center = np.array([(gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2])
    pred_center = np.array([(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2])

    
    center_error = np.linalg.norm(gt_center - pred_center)

    return center_error

def load_annotations(filepath):
    annotations = []
    with open(filepath, 'r') as file:
        for line in file:
            x, y, w, h = map(int, line.strip().split())
            annotations.append((x, y, x+w, y+h))
    return annotations


def compute_visual_similarity(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)


def get_visual_features(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[int(y1):int(y2), int(x1):int(x2)]
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()



