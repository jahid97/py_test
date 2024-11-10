import unittest
import os
import numpy as np
from segmentation_blood_ves import load_image, preprocess_image, segment_blood_vessels, blood_vessel_extract

class TestBloodVesselSegmentation(unittest.TestCase):

    def setUp(self):
        # Ensure this is a valid path to an image in the 'eye_blood' folder
        self.image_path = 'eye_blood1/10.png'
        
        # Check if the image file exists
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Test image not found at {self.image_path}")
        
        # Load the image for use in tests
        self.image = load_image(self.image_path)

    def test_load_image(self):
        """ Test if the image loads correctly """
        self.assertIsNotNone(self.image, "Failed to load the image.")

    def test_preprocess_image(self):
        """ Test if the preprocessing function returns an image of the expected type and dimensions """
        preprocessed_image = preprocess_image(self.image)
        self.assertIsInstance(preprocessed_image, np.ndarray, "Preprocessed image should be a numpy array.")
        self.assertEqual(preprocessed_image.shape, self.image.shape[:2], "Preprocessed image should be grayscale with original image dimensions.")

    def test_segment_blood_vessels(self):
        """ Test if segmentation outputs an image with binary values only """
        preprocessed_image = preprocess_image(self.image)
        segmented_image = segment_blood_vessels(preprocessed_image)
        unique_values = np.unique(segmented_image)
        self.assertTrue(set(unique_values).issubset({0, 255}), "Segmented image should contain binary values only (0 or 255).")

    def test_blood_vessel_extract(self):
        """ Test the overall blood vessel extraction pipeline """
        original_image, segmented_image = blood_vessel_extract(self.image_path)
        self.assertIsNotNone(original_image, "Original image is None.")
        self.assertIsNotNone(segmented_image, "Segmented image is None.")
        self.assertEqual(original_image.shape[0:2], segmented_image.shape, "Segmented image should match the original image dimensions.")

    def test_folder_images_processing(self):
        """ Test processing of all images in the folder """
        folder_path = 'eye_blood'
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                image, segmented_image = blood_vessel_extract(image_path)
                self.assertIsNotNone(image, f"Failed to load {image_name}.")
                self.assertIsNotNone(segmented_image, f"Failed to segment blood vessels in {image_name}.")

if __name__ == '__main__':
    unittest.main()
