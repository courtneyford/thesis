import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from lime.lime_image import LimeImageExplainer
from scipy.spatial import distance
from scipy.ndimage import convolve
from skimage.color import gray2rgb, rgb2gray
from typing import List, Tuple

MASK_IDX = 0
FEATURE_IDX = 1
IMG_ID_IDX = 3


def get_top_features(X, y, model, num_samples=500, num_features=3, max_per_image=5):
    """Compute the top features for each class in the dataset using LIME.
    *here features are unique for an image, so they are in fact super pixels.*
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The input data.
    y : ndarray, shape (n_samples,)
        The class labels.
    model : object
        The machine learning model to explain.
    num_samples : int, optional (default=1000)
        The number of samples to use when generating LIME explanations.
    num_features : int, optional (default=5)
        The number of features to select for each example.
        
    Returns
    -------
    class_top_features : dict
        A dictionary that maps each class label to a list of the top features
        for that class.
        list of (mask, image, score, imgid)
    """
    # Initialise dictionary to store top features for each class
    class_top_features = {}

    # Select all classes to examine
    classes_to_examine = np.unique(y)
    ids = np.arange(len(y))

    # Loop through the selected classes
    for c in classes_to_examine:
        # Filter the data by this class
        class_indices = np.where(y == c)[0]
        class_samples = X[class_indices]
        samples_id = ids[class_indices]
        assert np.all(y[class_indices] == c), "Error in samples selections"

        # Generate LIME explanations for each sample
        feature_importances = []
        for i, img in zip(samples_id, class_samples):
            explainer = lime_image.LimeImageExplainer(verbose=False)
            explanation = explainer.explain_instance( 
                    gray2rgb(img) ,
                    lambda imgs: model.predict_proba(np.array([rgb2gray(x).reshape(-1, 28, 28) for x in imgs])),
                     top_labels=1,
                                                      hide_color=0, num_samples=num_samples,
                                                      segmentation_fn=SegmentationAlgorithm('slic', n_segments=100, compactness=10, sigma=1))
            # Get the feature importance for the current class
            image_features = explanation.local_exp[explanation.top_labels[0]]
            image_features = sorted(image_features, key=lambda x: x[1], reverse=True)
            image_features = image_features[:max_per_image]


            # add feature encoding
            def encode_feature(idx):
                mask = explanation.segments == idx
                mask = mask[0]
                pixels = np.zeros_like(img[0])
                pixels[mask] = img[0][mask]
                if False:
                    plt.imshow(mask)
                    plt.figure()
                    plt.imshow(pixels)
                    plt.show()
                return (mask, pixels)

            # Compute additional information for each super pixels. 
            # (id of superpixel, score of superpixel, (mask, superpixels), id of image)
            image_features = [(f[0], f[1], encode_feature(f[0]), i) for f in image_features]

            # Store the feature importance in the list
            feature_importances.extend(image_features)

        # Sort the super pixels by their  importance and take the top num_features
        sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        top_features = [(f[2][0], f[2][1], f[1], f[3]) for f in sorted_features[:num_features]]

        # Store the top features in the dictionary
        class_top_features[c] = top_features
        
    return class_top_features


# Calculate Intersection over Union (IoU) between two masks
def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Determine if two masks are close based on IoU
def masks_are_close(mask1, mask2, iou_threshold=0.5):
    iou_score = iou(mask1, mask2)
    return iou_score > iou_threshold

# Calculate co-occurrence of superpixels within each image
def calculate_co_occurrence(class_top_features):
    co_occurrence_matrix_per_class = {}
    
    for class_label, top_features in class_top_features.items():
        n = len(top_features)
        co_occurrence_matrix = np.full((n, n), np.nan)  # Initialise with NaNs
        
        for i in range(n):
            for j in range(i+1, n):
                mask_i, _, _, imgid_i = top_features[i]
                mask_j, _, _, imgid_j = top_features[j]
                
                if imgid_i == imgid_j:
                    co_occurrence_matrix[i][j] = 1 
                    co_occurrence_matrix[j][i] = 1  

        co_occurrence_matrix_per_class[class_label] = co_occurrence_matrix

    return co_occurrence_matrix_per_class

def aggregate_superpixels(class_top_features):
    aggregated_features = {}

    # Calculate Co-occurrence matrix and set threshold
    co_occurrence_matrices = calculate_co_occurrence(class_top_features)
    co_occurrence_thresholds = {class_id: np.nanpercentile(matrix, 90) for class_id, matrix in co_occurrence_matrices.items()}

    for class_id, top_features in class_top_features.items():
        aggregated_features[class_id] = []
        added_masks = set()  # Keep track of masks that have already been added

        # Aggregate based on spatial proximity and score
        for i, (mask1, image1, score1, imgid1) in enumerate(top_features):
            if tuple(mask1.flatten()) in added_masks:
                continue  # Skip masks that have already been added

            aggregated = [score1]

            for j, (mask2, image2, score2, imgid2) in enumerate(top_features[i+1:], start=i+1):
                if masks_are_close(mask1, mask2):
                    aggregated.append(score2)

            mean_score = np.mean(aggregated)
            aggregated_features[class_id].append((mask1, image1, mean_score, imgid1))
            added_masks.add(tuple(mask1.flatten()))

        # Add masks that haven't been aggregated
        for mask1, image1, score1, imgid1 in top_features:
            if tuple(mask1.flatten()) not in added_masks:
                aggregated_features[class_id].append((mask1, image1, score1, imgid1))

        # Aggregate based on co-occurrence
        co_occurrence_matrix = co_occurrence_matrices[class_id]
        threshold = co_occurrence_thresholds[class_id]

        for i, (mask1, image1, score1, imgid1) in enumerate(top_features):
            if tuple(mask1.flatten()) in added_masks:
                continue  # Skip masks that have already been added
            
            aggregated = [score1]

            for j, (mask2, image2, score2, imgid2) in enumerate(top_features[i+1:], start=i+1):
                if co_occurrence_matrix[i, j] > threshold:
                    aggregated.append(score2)

            mean_score = np.mean(aggregated)
            aggregated_features[class_id].append((mask1, image1, mean_score, imgid1))
            added_masks.add(tuple(mask1.flatten()))

    return aggregated_features
