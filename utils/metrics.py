import numpy as np
import torch

def bbox_iou(box1, box2):
    """
    Compute IOU between two bounding boxes.
    Boxes are expected to be in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou

def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Compute area under PR curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

def non_max_suppression(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression to remove overlapping bounding boxes.
    """
    keep = []
    order = scores.argsort()[::-1]
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        iou = np.array([bbox_iou(boxes[i], boxes[j]) for j in order[1:]])
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=1):
    """
    Compute mAP for object detection results.
    """
    ap_per_class = []
    
    for c in range(num_classes):
        predictions = [box for box in pred_boxes if box[5] == c]
        targets = [box for box in true_boxes if box[5] == c]
        
        if len(predictions) == 0 and len(targets) == 0:
            ap_per_class.append(1.0)
            continue
        elif len(predictions) == 0 or len(targets) == 0:
            ap_per_class.append(0.0)
            continue
        
        # Sort predictions by confidence score
        predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
        
        TP = np.zeros(len(predictions))
        FP = np.zeros(len(predictions))
        
        num_targets = len(targets)
        detected_targets = []
        
        for i, pred in enumerate(predictions):
            best_iou = 0
            best_target = None
            
            for j, target in enumerate(targets):
                if j in detected_targets:
                    continue
                
                iou = bbox_iou(pred[:4], target[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_target = j
            
            if best_iou > iou_threshold:
                if best_target not in detected_targets:
                    TP[i] = 1
                    detected_targets.append(best_target)
                else:
                    FP[i] = 1
            else:
                FP[i] = 1
        
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        
        recalls = TP_cumsum / num_targets
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        
        ap = compute_ap(recalls, precisions)
        ap_per_class.append(ap)
    
    return np.mean(ap_per_class)