import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image

def mse(imageA, imageB):
    # Calculate the mean squared error between two images.
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compute_sift_matches(img1, img2):
    # Compute SIFT features and match them between two images.
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return len(good)

def compute_ahash(image_path):
    # Compute the average hash of an image.
    image = Image.open(image_path)
    h = imagehash.average_hash(image)
    return h

# List of image paths
image_paths = [f'images/3{i}.png' for i in range(0, 7 )]
print(f"running numbers on {image_paths}")
num_images = len(image_paths)

# Initialize sums
total_mse = 0
total_ssim = 0
total_sift_matches = 0
ahash_distances = []

# Compare each pair of images
for i in range(num_images):
    for j in range(i + 1, num_images):
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[j], cv2.IMREAD_GRAYSCALE)

        total_mse += mse(img1, img2)
        total_ssim += ssim(img1, img2)

        img1_color = cv2.imread(image_paths[i])
        img2_color = cv2.imread(image_paths[j])
        total_sift_matches += compute_sift_matches(img1_color, img2_color)

        # aHash comparison
        hash1 = compute_ahash(image_paths[i])
        hash2 = compute_ahash(image_paths[j])
        ahash_distances.append(hash1 - hash2)

# Compute averages
avg_mse = total_mse / (num_images * (num_images - 1) / 2)
avg_ssim = total_ssim / (num_images * (num_images - 1) / 2)
avg_sift_matches = total_sift_matches / (num_images * (num_images - 1) / 2)
avg_ahash_distance = np.mean(ahash_distances)

print(f"Average MSE: {avg_mse}")
print(f"Average SSIM: {avg_ssim}")
print(f"Average SIFT matches: {avg_sift_matches}")
print(f"Average aHash distance: {avg_ahash_distance}")

