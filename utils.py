import cv2
import numpy as np

def alignImagesNew(im1, im2):

    return 1





  # from: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
def alignImages(im1, im2, max_features=1000, good_match_percent=0.05):
    # Convert images to grayscale
    #im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1Gray = im1[..., 0]
    im2Gray = im2[..., 2]

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    print("---")

    # Sort matches by score
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    print(len(matches))
    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1Gray, h, (width, height))

    return im1Reg, h
