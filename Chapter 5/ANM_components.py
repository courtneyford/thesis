import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def compute_L_boundary(logits, true_label_index):
    """
    Compute the boundary loss for a given prediction to maximise overall loss.
    
    Parameters:
    logits (Tensor): The output logits from the model.
    true_label_index (int): The index of the true label.
    
    Returns:
    float: The modified boundary loss.
    """
    probabilities = tf.nn.softmax(logits, axis=1)
    decision_function_value = probabilities[0][true_label_index]
    SAFE_ZONE = (0.45, 0.55)
    
    if SAFE_ZONE[0] <= decision_function_value <= SAFE_ZONE[1]:
        # Penalise instances within the safe zone by returning a high loss
        return tf.constant(1.0)  # or another suitably high value
    else:
        # Reward instances outside the safe zone by returning a lower loss
        return tf.square(0.5 - decision_function_value)  # Closer to 0 for values away from 0.5

    return L_boundary.numpy()



def compute_L_feature(modified_image, rho, target_class, neighbor_class, prototypes, clustered_top_features, target_weight, neighbor_weight):
    modified_image_tensor = tf.convert_to_tensor(modified_image, dtype=tf.float32)
    
    # Retrieve the prototypes for the target and neighbour classes
    Prototype_neighbor = tf.convert_to_tensor(prototypes[neighbor_class], dtype=tf.float32)
    
    # Initialise the feature loss
    L_feature = tf.constant(0.0, dtype=tf.float32)
    
    # Iterate over the top features
    for feature_idx in range(3):
        # Retrieve the mask for the neighbour class's features
        M_neighbor_class = tf.cast(clustered_top_features[neighbor_class][feature_idx]['mask'], dtype=tf.float32)
        
        # Calculate the norm of the difference between the modified image and the neighbour's prototype, weighted by the mask
        term_neighbor = tf.norm(tf.multiply(M_neighbor_class, modified_image_tensor - Prototype_neighbor))
        
        # Apply differential weighting to the neighbour's feature term
        L_feature += neighbor_weight * term_neighbor
    
    # add a term for the target class's features with a lower weight
    if target_weight > 0:
        Prototype_target = tf.convert_to_tensor(prototypes[target_class], dtype=tf.float32)
        for feature_idx in range(3):
            M_target_class = tf.cast(clustered_top_features[target_class][feature_idx]['mask'], dtype=tf.float32)
            term_target = tf.norm(tf.multiply(M_target_class, modified_image_tensor - Prototype_target))
            L_feature -= target_weight * term_target  # Note the subtraction, reducing the influence
    
    # Multiply by rho to scale the feature loss
    L_feature *= rho
    return L_feature.numpy()

        
def compute_L_similarity(modified_image_tensor, target_class, neighbor_class, prototypes):
    """
    Compute the similarity loss for a modified image.

    Parameters:
    modified_image_tensor (Tensor or np.ndarray): The modified image tensor.
    target_class (int): The index of the target class.
    neighbor_class (int): The index of the neighbor class.
    prototypes (dict): A dictionary containing class prototypes.

    Returns:
    Tensor: The similarity loss as a tensor.
    """
    # Convert inputs to tensors if they're not already
    modified_image_tensor = tf.convert_to_tensor(modified_image_tensor, dtype=tf.float32)
    Prototype_target = tf.convert_to_tensor(prototypes[target_class], dtype=tf.float32)
    Prototype_neighbor = tf.convert_to_tensor(prototypes[neighbor_class], dtype=tf.float32)

    similarity_with_target = tf.norm(modified_image_tensor - Prototype_target)
    similarity_with_neighbor = tf.norm(modified_image_tensor - Prototype_neighbor)
    L_similarity = similarity_with_neighbor - similarity_with_target

    return L_similarity


def compute_L_directionality(modified_image_tensor, original_image_tensor, delta, model, target_class, neighbor_class, prototypes):
    """
    Compute the directionality loss for a modified image.

    Parameters:
    modified_image_tensor (Tensor): The modified image tensor.
    original_image_tensor (Tensor): The original image tensor before modification.
    delta (float): The weight factor for directionality loss.
    target_class (int): The index of the target class.
    neighbor_class (int): The index of the neighbor class.
    prototypes (dict): A dictionary containing class prototypes.

    Returns:
    float: The directionality loss.
    """
    with tf.GradientTape() as tape:
        tape.watch(modified_image_tensor)
        L_similarity = compute_L_similarity(modified_image_tensor, target_class, neighbor_class, prototypes)

    grads_similarity = tape.gradient(L_similarity, modified_image_tensor)
    delta_tensor = tf.convert_to_tensor(delta, dtype=tf.float32)
    L_directionality = delta_tensor * tf.reduce_sum(grads_similarity * (modified_image_tensor - original_image_tensor))
    return L_directionality.numpy()

