

import hausdorff
import numpy as np 

import cv2
def transform_image_data(predict: np.ndarray, label: np.ndarray):
    predict = predict.astype(np.bool_).astype(np.int_)
    label = label.astype(np.bool_).astype(np.int_)
    return predict, label

def dice_coef(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict * label).sum()
    return (2. * intersection + epsilon) / (predict.sum() + label.sum() + epsilon)

def iou_score(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict & label).sum()
    union = (predict | label).sum()
    return (intersection + epsilon) / (union + epsilon)

def sensitivity(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict * label).sum()
    return (intersection + epsilon) / (label.sum() + epsilon)

def ppv(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict * label).sum()
    return (intersection + epsilon) / (predict.sum() + epsilon)

def hd95(predict: np.ndarray, label: np.ndarray, distance="euclidean"):
    predict, label = transform_image_data(predict, label)
    predict = predict.flatten()[..., None]
    label = label.flatten()[..., None]
    distance = hausdorff.hausdorff_distance(predict, label, distance=distance)
    return distance * 0.95

if __name__ =="__main__":
    gt_image = cv2.imread('./gt.png', 0)
    kmeans_image = cv2.imread('./kmeans.mask.png', 0)
    # 单个指标
    dice_coef(kmeans_image, gt_image)
    
    # 所有指标
    metrics = {
        'Dice': dice_coef,
        'IoU': iou_score,
        'Sensitivity': sensitivity,
        'PPV': ppv,
        'HD95': hd95
    }

    for metric_name in metrics:
        score = round(metrics[metric_name](predict, label), 3)
        print('{}:{}'.format(metric_name, score))

