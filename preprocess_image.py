import cv2
import numpy as np
import os

DATASET_PATH = "../data/train/images"  

def load_image(file_name):
    """Load an image from the dataset."""
    image_path = os.path.join(DATASET_PATH, file_name)
    image = cv2.imread(image_path)  
    if image is None:
        raise FileNotFoundError(f"Image {file_name} not found in {DATASET_PATH}")
    return image

def preprocess_image(image, target_size=(1024, 1024)):
    """Resize and normalize the image."""
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0
    return image_resized, image_normalized

def adjust_bounding_boxes(annotations, original_size, new_size=(1024, 1024)):
    """Adjust bounding box coordinates to fit the resized image."""
    original_width, original_height = original_size
    new_width, new_height = new_size
    annotations_resized = []
    for annotation in annotations:
        x_center, y_center, width, height = annotation[1:]
        x_center_resized = (x_center * original_width) / new_width
        y_center_resized = (y_center * original_height) / new_height
        width_resized = (width * original_width) / new_width
        height_resized = (height * original_height) / new_height
        annotations_resized.append([annotation[0], x_center_resized, y_center_resized, width_resized, height_resized])
    return annotations_resized


def display_resized_image_for_screen(image, target_size=(800, 800), title="Processed Image"):
    """Resize image for screen and display."""
    image_resized_for_display = cv2.resize(image, target_size)  
    cv2.imshow(title, image_resized_for_display)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sample_file = "IMG_2274_jpeg_jpg.rf.2f319e949748145fb22dcb52bb325a0c.jpg"  
    image = load_image(sample_file)
    image_resized, image_normalized = preprocess_image(image)
    display_resized_image_for_screen(image_resized, target_size=(800, 800), title="Resized Image for Screen")
    print(f"Original image shape: {image.shape}")
    print(f"Resized image shape: {image_resized.shape}")
    annotations = [
        [0, 0.5, 0.5, 0.2, 0.2],  
        [1, 0.3, 0.3, 0.1, 0.1]
    ]
    adjusted_annotations = adjust_bounding_boxes(annotations, image.shape[1:], (1024, 1024))
    print("Adjusted bounding boxes:")
    for annotation in adjusted_annotations:
        print(annotation)
