import cv2
import numpy as np
import matplotlib.pyplot as plt

# HomeTask 3A
def stitch_images(img1, img2):
    """Stitches two images together using SIFT feature matching and homography transformation."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Use Brute-Force matcher to find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test to filter good matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good_matches) < 10:
        print("Not enough matches!")
        return None
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography matrix
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    # Warp the second image to align with the first
    stitched_img = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    stitched_img[:img1.shape[0], :img1.shape[1]] = img1
    
    return stitched_img

# HomeTask 3B

def blend_images(img1, img2):
    """Applies blending to reduce seams in the stitched image."""
    mask = np.zeros_like(img1, dtype=np.uint8)
    mask[:, :img1.shape[1]//2] = 255
    blended = cv2.seamlessClone(img1, img2, mask, (img1.shape[1]//2, img1.shape[0]//2), cv2.NORMAL_CLONE)
    return blended

# Load images
img0 = cv2.imread('D:\Computer Vision\Assignments\HomeTask#3-ImageStitching\stitch0.jpg')
img1 = cv2.imread('D:\Computer Vision\Assignments\HomeTask#3-ImageStitching\stitch1.jpg')
img2 = cv2.imread('D:\Computer Vision\Assignments\HomeTask#3-ImageStitching\stitch2.jpg')

# Display original images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
axs[0].set_title('Image 0')
axs[0].axis('off')
axs[1].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axs[1].set_title('Image 1')
axs[1].axis('off')
axs[2].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axs[2].set_title('Image 2')
axs[2].axis('off')
plt.show()

# Stitch images sequentially
stitched_1_2 = stitch_images(img1, img2)
stitch_final = stitch_images(img0, stitched_1_2)

# Display warped images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(stitched_1_2, cv2.COLOR_BGR2RGB))
axs[0].set_title('Stitched Image 1 & 2')
axs[0].axis('off')
axs[1].imshow(cv2.cvtColor(stitch_final, cv2.COLOR_BGR2RGB))
axs[1].set_title('Final Stitched Image')
axs[1].axis('off')
plt.show()

# Apply blending
final_result = blend_images(img0, stitch_final)

# Display final blended image
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Final Blended Image')
plt.show()

# Save result
cv2.imwrite('stitched_output.jpg', final_result)
