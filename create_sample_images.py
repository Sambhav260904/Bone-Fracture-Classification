import cv2
import numpy as np
import os

def create_sample_images(directory, count=5):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(count):
        # Generate a random image
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        # Add text to the image
        cv2.putText(image, f'Image {i + 1}', (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 2)
        # Save the image
        cv2.imwrite(os.path.join(directory, f'input_image_{i + 1}.jpg'), image)

# Specify the directory and number of images to create
if __name__ == "__main__":
    create_sample_images('sample_data', count=10)
