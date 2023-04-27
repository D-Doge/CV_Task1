import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

POINTS_FROM_POSTER_CAT = np.array([[200, 150], [300, 150],
                          [300, 250], [200, 250]])

POINTS_FROM_POSTER_CAT_OFFSET = np.array([[1200, 1150], [1300, 1150],
                          [1300, 1250], [1200, 1250]])

IMAGE_DIR = "images/"

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def raid_to_deg(angle):
    return angle * 180.0/ math.pi

def angle_between_deg(v1, v2):
    return raid_to_deg(angle_between(v1, v2))

def calc_alpha_1(h1, h2, p1, p4):
    p1 = np.array(p1)
    p4 = np.array(p4)

    h1 = np.array(h1)
    h2 = np.array(h2)

    v1 = (p4 - p1) - (p1 - p1)
    v2 = (h2 - h1) - (h1 - h1)
    
    return angle_between_deg(v1, v2)

def calc_alpha_2(h1, h2, p2, p3):
    p2 = np.array(p2)
    p3 = np.array(p3)

    h1 = np.array(h1)
    h2 = np.array(h2)

    v1 = (p3 - p2) - (p2 - p2)
    v2 = (h2 - h1) - (h1 - h1)
    
    return angle_between_deg(v1, v2)

def calc_beta_1(v1, v2, p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    v1 = np.array(v1)
    v2 = np.array(v2)

    vec1 = (p2 - p1) - (p1 - p1)
    vec2 = (v2 - v1) - (v1 - v1)
    
    return angle_between_deg(vec1, vec2)

def calc_beta_2(v1, v2, p3, p4):
    p3 = np.array(p3)
    p4 = np.array(p4)

    v1 = np.array(v1)
    v2 = np.array(v2)

    vec1 = (p3 - p4) - (p4 - p4)
    vec2 = (v2 - v1) - (v1 - v1)
    
    return angle_between_deg(vec1, vec2)

def calc_all_angles(p1, p2, p3, p4, h1, h2, v1, v2):
    alpha1 = calc_alpha_1(h1, h2, p1, p4)
    alpha2 = calc_alpha_2(h1, h2, p2, p3)
    beta1 = calc_beta_1(v1, v2, p1, p2)
    beta2 = calc_beta_2(v1, v2, p3, p4)
    
    return alpha1, alpha2, beta1, beta2

def transfrom_point(point, h):
    # Define the point you want to transform
    point = np.array(point)
    point = np.append(point, 1)

    # Multiply the homography matrix by the point
    transformed_point = np.dot(h, point)

    # Divide by the third component to obtain Cartesian coordinates
    x_transformed = transformed_point[0] / transformed_point[2]
    y_transformed = transformed_point[1] / transformed_point[2]

    return (x_transformed, y_transformed)


points = [["niedermaier_tobias_32900_20221115_113319.jpg", (1436, 1760), (1439, 2125), (2026, 2109), (2023, 1763), (1436, 1503), (2023, 1498), (750, 1760),  (778, 2125)],
["niedermaier_tobias_32900_20221115_113328.jpg", (1749, 1874), (1742, 2062), (2058, 2057), (2046, 1868), (1749, 1737), (2046, 1726), (1380, 1874), (1385, 2062)],
["niedermaier_tobias_32900_20221115_113340.jpg", (1693, 1769), (1719, 2370), (2198, 2282), (2193, 1793), (1963, 1344), (2193, 1450), (701, 1769),  (747, 2370)],
["niedermaier_tobias_32900_20221115_113346.jpg", (953, 1603),  (978, 2111),  (1630, 2080), (1624, 1629),  (953, 1235), (1624, 1318), (1955, 1603), (1952, 2111)],
["niedermaier_tobias_32900_20221115_113356.jpg", (1599, 1789), (1602, 2099), (1985, 2099), (1982, 1789), (1599, 1583), (1982, 1545), (2311, 1789), (2306, 2099)],
["niedermaier_tobias_32900_20221115_113401.jpg", (1513, 1361), (1526, 1661), (1924, 1671), (1930, 1352), (1513, 1156), (1930, 1133), (2226, 1361), (2231, 1661)],
["niedermaier_tobias_32900_20221115_113424.jpg", (1065, 1633), (1110, 2353), (1949, 2466), (1927, 1460), (1065, 1104), (1927, 744),  (2922, 1633), (2914, 2353)],
["niedermaier_tobias_32900_20221115_113437.jpg", (395, 2692),  (586, 4622),  (1843, 3376), (1805, 2292), (395, 1099),  (1805, 1527), (2124, 2692), (2199, 4622)],
["niedermaier_tobias_32900_20221115_113440.jpg", (200, 606),   (439, 3037),  (1968, 2424), (1961, 1237), (1805, 208), (2669, 960), (2373, 721), (2336, 3502)]]


#Feed everypoint into backwards H matrix
for point in points:
    image = point[0]

    filename = image[25:]
    img_path = os.path.join(IMAGE_DIR, filename)
    #read image
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    poster = cv2.imread('cat.jpg')

    augmentent_frame = cv2.imread("out_5/" + "niedermaier_tobias_32900_" + filename)

    #Load the dictionary that was used to generate the markers.
    #We seem to have 6X6 Arucos
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters()

    # Detect the markers in the image
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, reject = detector.detectMarkers(frame)

    if(len(corners) == 0):
        continue

    #Calculate Homography
    h, _ = cv2.findHomography(corners[0], POINTS_FROM_POSTER_CAT_OFFSET) 

    point = point[1:]

    for i, p in enumerate(point):
        point[i] = transfrom_point(p, h)


    #Wrape the poster
    #warped_poster = cv2.warpPerspective(augmentent_frame, h, (frame.shape[1] * 2 ,frame.shape[0] * 2))

    #temp = "out_test/" + "niedermaier_tobias_32900_" + filename

    #cv2.imwrite(temp, warped_poster)



    if(image != "niedermaier_tobias_32900_20221115_113437.jpg"):
        print(image[34:40], calc_all_angles(point[0], point[1], point[2], point[3], point[4], point[5], point[6], point[7]))
    else:
        alpha1, _, beta1, beta2 = calc_all_angles(point[0], point[1], point[2], point[3], point[4], point[5], point[6], point[7])
        _, alpha2, _, _ = calc_all_angles(point[0], transfrom_point((890, 4619), h), point[2], point[3], point[4], point[5], point[6], point[7])
        print(image[34:40], alpha1, alpha2, beta1, beta2)





