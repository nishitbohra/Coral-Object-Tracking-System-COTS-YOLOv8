import cv2
import numpy as np
import os
import random

DATASET_PATH = "../data/train/images"  
def load_image(file_name):
    """Load an image from the dataset."""
    image_path = os.path.join(DATASET_PATH, file_name)
    image = cv2.imread(image_path)  
    if image is None:
        raise FileNotFoundError(f"Image {file_name} not found in {DATASET_PATH}")
    return image

def random_flip(image, annotations):
    """Randomly flip the image and adjust bounding boxes accordingly."""
    if random.random() > 0.5:  
        image = cv2.flip(image, 1)  
        for annotation in annotations:
            annotation[1] = 1 - annotation[1]  
    return image, annotations

def random_rotation(image, annotations, max_angle=30):
    """Randomly rotate the image and adjust bounding boxes accordingly."""
    angle = random.uniform(-max_angle, max_angle)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h))
    return image, annotations

def random_brightness_contrast(image, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5)):
    """Randomly adjust brightness and contrast."""
    alpha = random.uniform(*contrast_range)  
    beta = random.uniform(*brightness_range)  
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

def augment_data(image, annotations):
    """Apply random augmentations to the image and bounding boxes."""
    image, annotations = random_flip(image, annotations)
    image, annotations = random_rotation(image, annotations)
    image = random_brightness_contrast(image)
    
    return image, annotations

def display_image(image, title="Augmented Image"):
    """Display the processed image."""
    cv2.imshow(title, image)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sample_file = "IMG_2274_jpeg_jpg.rf.2f319e949748145fb22dcb52bb325a0c.jpg"  
    image = load_image(sample_file)
    annotations = [
        [0, 0.5, 0.5, 0.2, 0.2],  
        [1, 0.3, 0.3, 0.1, 0.1]   
    ]
    image_augmented, annotations_augmented = augment_data(image, annotations)
    display_image(image_augmented, title="Augmented Image")
    print("Adjusted annotations after augmentation:")
    for annotation in annotations_augmented:
        print(annotation)
