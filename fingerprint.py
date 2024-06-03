import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from skimage.morphology import convex_hull_image, erosion, square

def preprocess_fingerprint(image, visualize):
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError("Image not loaded correctly. Please check the file path and format.")
    
    # Convert to grayscale if not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        raise ValueError("Unsupported image format. Please provide a grayscale or BGR image.")
        
    
    if visualize:
        plt.subplot(1, 4, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('1. Grayscale')
        plt.axis('off')

        
    normed = (gray - np.mean(gray)) / (np.std(gray))


    blk_size = 9
    C = 10
    # Apply thresholding to binarize the image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk_size, C)
    img_erode = cv2.erode(binary, (10,10), iterations = 3)
    img_dil = cv2.dilate(img_erode, (10,10), iterations = 4)

    if visualize:
        plt.subplot(1, 4, 2)
        plt.imshow(img_dil, cmap='gray')
        plt.title('2. binary')
        plt.axis('off')
    
    # Invert the binary image
    inverted_binary = cv2.bitwise_not(img_dil)
    
    if visualize:
        plt.subplot(1, 4, 3)
        plt.imshow(inverted_binary, cmap='gray')
        plt.title('3. Inverted')
        plt.axis('off')
    
    # Skeletonize the image
    skeleton = skeletonize(inverted_binary // 255)  # Convert to boolean for skeletonize
    skeleton = (skeleton * 255).astype(np.uint8)    # Convert back to uint8
    
    if visualize:
        plt.subplot(1, 4, 4)
        plt.imshow(skeleton, cmap='gray')
        plt.title('4. Skeleton')
        plt.axis('off')

    if visualize:
        plt.tight_layout()
        plt.show()

    return skeleton

def find_minutiae(skeleton):
    endpoints = []
    bifurcations = []
    minutiaeTerm = np.zeros(skeleton.shape)
    
    # Iterate through the image to find minutiae points
    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            if skeleton[i, j] == 255:
                # Extract 3x3 window around the current pixel
                window = skeleton[i-1:i+2, j-1:j+2]
                
                # Count the number of white pixels (neighbors)
                count = np.sum(window == 255)
                
                if count == 2:
                    minutiaeTerm[i][j] = 1
                elif count > 3:
                    bifurcations.append((i, j))
    
    # 마스크 이미지의 볼록 껍질 및 침식 처리
    mask = skeleton * 255
    mask = convex_hull_image(mask > 0)
    mask = erosion(mask, square(20))  # Structuing element for mask erosion = square(5)

    # 종료점 배열과 마스크의 논리적 AND 연산
    minutiaeTerm = np.uint8(mask) * minutiaeTerm

    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            if minutiaeTerm[i][j] == 1:
                endpoints.append((i, j))

    return np.array(endpoints), np.array(bifurcations)

def plot_minutiae_points(image, endpoints, bifurcations):
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Plot the image
    plt.imshow(image, cmap='gray')
    plt.title('Minutiae Points')
    
    # Plot endpoints with red dots
    for point in endpoints:
        plt.plot(point[1], point[0], 'ro', markersize=3)
    
    # Plot bifurcations with blue dots
    for point in bifurcations:
        plt.plot(point[1], point[0], 'bo', markersize=3)
    
    plt.axis('off')
    plt.show()    

def process_fingerprint_image(image, visualize=False):
    skeleton = preprocess_fingerprint(image, visualize)
    endpoints, bifurcations = find_minutiae(skeleton)
    if(visualize):
        plot_minutiae_points(skeleton, endpoints, bifurcations)
    return endpoints, bifurcations


def match_finger(feat_query, feat_train, distance_threshold):
    match_num = 0

    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(feat_query, feat_train), key= lambda match:match.distance)

	# Calculate score
    match_score = 0
    for match in matches:
        if match.distance < distance_threshold:
            match_score += match.distance
            match_num += 1
            
    return match_score, match_num

def match_score(feat_query, feat_train, distance_threshold):

    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(feat_query, feat_train), key= lambda match:match.distance)

	# Calculate score
    match_score = 0
    for match in matches:
        if match.distance < distance_threshold:
            match_score += match.distance
            
    return match_score
