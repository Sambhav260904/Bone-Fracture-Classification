# its show the output window

# import os
# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from skimage.feature import local_binary_pattern, hog
# from skimage import exposure
# import matplotlib.pyplot as plt

# # Function to draw grid on an image
# def draw_grid(image_path, output_path, rows, cols, line_color=(255, 0, 0), line_width=2):
#     image = Image.open(image_path)
#     draw = ImageDraw.Draw(image)
#     width, height = image.size
#     cell_width = width / cols
#     cell_height = height / rows
    
#     for i in range(1, cols):
#         x = i * cell_width
#         draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
    
#     for i in range(1, rows):
#         y = i * cell_height
#         draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

#     # Convert to RGB if image is in RGBA mode
#     if image.mode == 'RGBA':
#         image = image.convert('RGB')

#     image.save(output_path)

# # Function to compute LBP
# def compute_lbp(image_path, output_path, radius=3, n_points=24):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     lbp = local_binary_pattern(image, n_points, radius, method='uniform')
#     cv2.imwrite(output_path, lbp)
#     plt.imshow(lbp, cmap='gray')
#     plt.title('Local Binary Patterns (LBP)')
#     plt.show()

# # Function to compute HOG
# def compute_hog(image_path, output_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#     cv2.imwrite(output_path, hog_image_rescaled)
#     plt.imshow(hog_image_rescaled, cmap='gray')
#     plt.title('Histogram of Oriented Gradients (HOG)')
#     plt.show()

# # Function to compute LOOP
# def compute_loop(image_path, output_path, radius=1):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     rows, cols = image.shape
#     offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
#     loop_image = np.zeros((rows, cols), dtype=np.uint8)
    
#     for r in range(radius, rows - radius):
#         for c in range(radius, cols - radius):
#             center_pixel = image[r, c]
#             code = 0
#             for idx, (dr, dc) in enumerate(offsets):
#                 neighbor_pixel = image[r + dr, c + dc]
#                 if neighbor_pixel >= center_pixel:
#                     code |= (1 << idx)
#             loop_image[r, c] = code
    
#     cv2.imwrite(output_path, loop_image)
#     plt.imshow(loop_image, cmap='gray')
#     plt.title('Local Optimal Oriented Pattern (LOOP)')
#     plt.show()

# # Main function to process images
# def process_images(image_paths, output_dir, rows, cols):
#     for image_path in image_paths:
#         image_name = os.path.splitext(os.path.basename(image_path))[0]
        
#         # Draw grid
#         grid_output_path = os.path.join(output_dir, f"{image_name}_grid.jpg")
#         draw_grid(image_path, grid_output_path, rows, cols)
        
#         # Compute LBP
#         lbp_output_path = os.path.join(output_dir, f"{image_name}_lbp.jpg")
#         compute_lbp(image_path, lbp_output_path)
        
#         # Compute HOG
#         hog_output_path = os.path.join(output_dir, f"{image_name}_hog.jpg")
#         compute_hog(image_path, hog_output_path)
        
#         # Compute LOOP
#         loop_output_path = os.path.join(output_dir, f"{image_name}_loop.jpg")
#         compute_loop(image_path, loop_output_path)

# # Main function to iterate through the dataset
# def main():
#     dataset_dir = r'C:\Users\User\Desktop\ImageProcessingProject\Bone Break Classification\Bone Break Classification'  # Update this path
#     output_dir = 'output'
#     os.makedirs(output_dir, exist_ok=True)

#     # Iterate through each fracture type
#     for fracture_type in os.listdir(dataset_dir):
#         fracture_path = os.path.join(dataset_dir, fracture_type)
#         for category in ['Train', 'Test']:
#             category_path = os.path.join(fracture_path, category)
#             if os.path.exists(category_path):  # Check if the path exists
#                 image_paths = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]
#                 process_images(image_paths, output_dir, rows=4, cols=4)
#             else:
#                 print(f"Category path does not exist: {category_path}")

# if __name__ == "__main__":
#     main()







import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import local_binary_pattern, hog
from skimage import exposure

# Function to draw grid on an image
def draw_grid(image_path, rows, cols):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    cell_width = width / cols
    cell_height = height / rows
    
    for i in range(1, cols):
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill=(255, 0, 0), width=2)
    
    for i in range(1, rows):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill=(255, 0, 0), width=2)

    return image

# Function to compute LBP
def compute_lbp(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp

# Function to compute HOG
# def compute_hog(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
#     return hog_image


# Function to compute HOG
def compute_hog(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    
    # Rescale the HOG image to 0-255 for proper visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_image_rescaled = (hog_image_rescaled * 255).astype(np.uint8)  # Convert to 8-bit image
    
    return hog_image_rescaled




# Function to compute LOOP
def compute_loop(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = image.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    loop_image = np.zeros((rows, cols), dtype=np.uint8)
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center_pixel = image[r, c]
            code = 0
            for idx, (dr, dc) in enumerate(offsets):
                neighbor_pixel = image[r + dr, c + dc]
                if neighbor_pixel >= center_pixel:
                    code |= (1 << idx)
            loop_image[r, c] = code
    
    return loop_image

# Main function to process images
def process_images(image_paths, output_dir, fracture_type):
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            continue
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Draw grid
        grid_image = draw_grid(image_path, rows=4, cols=4)
        grid_output_path = os.path.join(output_dir, f"{fracture_type} {image_name} grid.jpg")
        grid_image.save(grid_output_path)

        # Compute and save LBP
        lbp_image = compute_lbp(image_path)
        lbp_output_path = os.path.join(output_dir, f"{fracture_type} {image_name} lbp.jpg")
        cv2.imwrite(lbp_output_path, lbp_image)

        # Compute and save HOG
        hog_image = compute_hog(image_path)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_output_path = os.path.join(output_dir, f"{fracture_type} {image_name} hog.jpg")
        cv2.imwrite(hog_output_path, hog_image_rescaled)

        # Compute and save LOOP
        loop_image = compute_loop(image_path)
        loop_output_path = os.path.join(output_dir, f"{fracture_type} {image_name} loop.jpg")
        cv2.imwrite(loop_output_path, loop_image)

# Main function to organize processing
def main():
    dataset_dir = r'C:\Users\User\Desktop\ImageProcessingProject\Bone Break Classification\Bone Break Classification'  # Update this path if necessary
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each fracture type
    for fracture_type in os.listdir(dataset_dir):
        fracture_path = os.path.join(dataset_dir, fracture_type)
        for category in ['Train', 'Test']:
            category_path = os.path.join(fracture_path, category)
            if os.path.exists(category_path):
                image_paths = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]
                if not image_paths:
                    print(f"No images found in: {category_path}")
                    continue
                process_images(image_paths, output_dir, fracture_type)
            else:
                print(f"Category path does not exist: {category_path}")

if __name__ == "__main__":
    main()
