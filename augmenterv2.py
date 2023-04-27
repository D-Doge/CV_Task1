import cv2
import matplotlib.pyplot as plt
import numpy as np

POINTS_FROM_POSTER_CAT = np.array([[200, 150], [300, 150],
                          [300, 250], [200, 250]])

#read image
frame = cv2.imread("images/20221115_113319.jpg", cv2.IMREAD_COLOR)
poster = cv2.imread('cat.jpg')

#Load the dictionary that was used to generate the markers.
#We seem to have 6X6 Arucos
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
 
# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters()
 

# Detect the markers in the image
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, reject = detector.detectMarkers(frame)

#Calculate Homography
h, _ = cv2.findHomography(POINTS_FROM_POSTER_CAT, corners[0]) 

#Wrape the poster
warped_poster = cv2.warpPerspective(poster, h, (frame.shape[1],frame.shape[0]))


# Create a mask where black pixels in the foreground image are white, and all other pixels are black
mask = cv2.cvtColor(warped_poster, cv2.COLOR_BGR2GRAY)
mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.show()

# Need to understand this -----------------------------------------------
# Erode the mask to not copy the boundary effects from the warping
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask = cv2.erode(mask, element, iterations=3)

# Copy the mask into 3 channels.
warped_poster = warped_poster.astype(float)
mask3 = np.zeros_like(warped_poster)
for i in range(0, 3):
    mask3[:,:,i] = mask/255

# Copy the warped image into the original frame in the mask region.
warped_image_masked = cv2.multiply(warped_poster, mask3)
frame_masked = cv2.multiply(frame.astype(float), 1-mask3)
im_out = cv2.add(warped_image_masked, frame_masked)

new_frame = im_out.astype(np.uint8)





plt.imshow(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
plt.show()