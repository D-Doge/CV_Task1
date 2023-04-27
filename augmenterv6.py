import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

POINTS_FROM_POSTER_CAT = np.array([[200, 150], [300, 150],
                          [300, 250], [200, 250]])

IMAGE_DIR = "images/"


for filename in os.listdir(IMAGE_DIR):


    img_path = os.path.join(IMAGE_DIR, filename)
    #read image
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    poster = cv2.imread('cat.jpg')

    #Load the dictionary that was used to generate the markers.
    #We seem to have 6X6 Arucos
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters()
    
    start_time = time.time()
    # Detect the markers in the image
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, reject = detector.detectMarkers(frame)

    if(len(corners) == 0):
        continue

    #Calculate Homography
    h, _ = cv2.findHomography(POINTS_FROM_POSTER_CAT, corners[0]) 



    # Remove (nearly) black Pixels before wrapping, this way we dont cut out anything while masking
    threshold = [6, 6, 6]
    mask = (poster < threshold).all(axis=2)
    poster[mask] = threshold

    #Wrape the poster
    warped_poster = cv2.warpPerspective(poster, h, (frame.shape[1],frame.shape[0]))


    # Create a mask where black pixels in the foreground image are white, and all other pixels are black
    mask = cv2.cvtColor(warped_poster, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]


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

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time)

    #plt.imshow(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
    #plt.show()

    temp = "out_5/" + "niedermaier_tobias_32900_" + filename

    cv2.imwrite(temp, new_frame)