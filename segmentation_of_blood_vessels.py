import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image at {image_path}")
    return image

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 5)
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
    clahe_image = clahe.apply(blurred_image)
    return clahe_image

def segment_blood_vessels(preprocessed_image):
    thresh_image = cv2.adaptiveThreshold(preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel, iterations=1)
    morph_image = cv2.erode(morph_image, kernel, iterations=1)
    morph_image = cv2.dilate(morph_image, kernel, iterations=1)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel, iterations=1)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morph_image

def blood_vessel_extract(image_path):
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image)
    segmented_image = segment_blood_vessels(preprocessed_image)
    return image, segmented_image

def main():
    folder_path = 'eye_blood1'
    images_data = []  # List to store tuples of (original_image, segmented_image)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
            continue

        try:
            image, lesion_image = blood_vessel_extract(image_path)
            images_data.append((image, lesion_image))

        except ValueError as e:
            print(e)

    # Display all images in one window using a grid layout
    num_images = len(images_data)
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 6 * num_images))

    for i, (original_image, segmented_image) in enumerate(images_data):
        axes[i, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original Eye Fundus Image {i+1}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(segmented_image, cmap='gray')
        axes[i, 1].set_title(f"Segmented Blood Vessels {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
