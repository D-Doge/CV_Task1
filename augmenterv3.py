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
mask = cv2.GaussianBlur(mask, (5, 5), 0)
mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.show()

# Invert the mask so that black pixels become zero, and non-black pixels become 1
mask_inv = cv2.bitwise_not(mask)

# Blend the foreground and background images together using the mask
foreground_masked = cv2.bitwise_and(warped_poster, warped_poster, mask=mask)
background_masked = cv2.bitwise_and(frame, frame, mask=mask_inv)
new_frame = cv2.addWeighted(foreground_masked, 1, background_masked, 1, 0)


plt.imshow(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
plt.show()