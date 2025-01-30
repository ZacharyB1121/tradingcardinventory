import cv2
import numpy as np
from itertools import combinations

#Open out webcam
CAMERA = cv2.VideoCapture(0, cv2.CAP_DSHOW)
CAMERA.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CAMERA.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
CAMERA.set(cv2.CAP_PROP_FOCUS, 200)

#Capture an image
returncode, frame = CAMERA.read()
if not returncode:
    print("Camera error")
    exit(1)

#Set up input and outputs
pre_image = frame.copy()

cv2.imwrite("pre_image.jpg", pre_image)

#Set up data needed to find markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

aruco_cropped_image = None

#Use OpenCV to find the markers
marker_bounding_boxes, ids, _ = cv2.aruco.detectMarkers(pre_image, aruco_dict, parameters=parameters)

# Make sure we found our 4 markers
if ids is not None and len(ids) == 4:  
    
    #We want to...
    # Find the markers
    # Get their location
    # And crop the image down to the area inside the markers, without including any of the marker
    centers = []
    max_width = 0
    max_height = 0
    
    for box in marker_bounding_boxes:
        box_croners = box[0]
        center_x = int(box_croners[:, 0].mean()) #Sum all the X coordinates and get the average for the center X
        center_y = int(box_croners[:, 1].mean()) # Do the same for the Y

        width = int(box_croners[:, 0].max() - box_croners[:, 0].min()) #Find the two points most opposite on the X axis, and subtract them to get the width
        height = int(box_croners[:, 1].max() - box_croners[:, 1].min()) #Do the same for the Y

        if width > max_width:
            max_width = width #Keep track of the widest marker's width
        if height > max_height:
            max_height = height #Keep track of the tallest marker's height

        centers.append((center_x, center_y))

    # We'll now basically find the x and y of the markers, and then offset by the max width and height so we are inside the markers
    min_x = min(centers, key=lambda x: x[0])[0] + max_width / 2
    max_x = max(centers, key=lambda x: x[0])[0] - max_width / 2
    min_y = min(centers, key=lambda x: x[1])[1] + max_height / 2
    max_y = max(centers, key=lambda x: x[1])[1] - max_height / 2

    # Defind the four inside corners of the markers
    centers_np = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ], dtype="float32")

    #Create a rectangle from the corners
    rect = cv2.convexHull(centers_np)
    rect = np.array(rect, dtype="float32")

    # Define the dimensions of the cropped area (desired output size)
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    # This allows us to make the image "square" and auto rotate it to be straight
    M = cv2.getPerspectiveTransform(rect, dst)

    # Perform the warp
    warped = cv2.warpPerspective(pre_image, M, (width, height))
    aruco_cropped_image = warped.copy()
else:
    print("No Aruco markers found")
    exit(1)


cv2.imwrite("aruco_image.jpg", aruco_cropped_image)
#By now we have just the inside of our markers, it time to find the card and essentially do the same thing

#We'll convert the image to grayscale and then use a threshold to make it black and white, and blur it
#   The idea being the card is colorful and will result in a black rectange surrounding our card
#   The blur helps smooth out the image, and account for noise
gray = cv2.cvtColor(aruco_cropped_image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 130, 255, 0) #130 is the threshold value, 0-255, I found it via experimentation
blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

#OpenCV can find the edges of what is hopefully our card
edges = cv2.Canny(blurred, 50, 150)

cv2.imwrite("edges.jpg", edges)

#Find contours of the edges, the idea being the contour will "outline" the card for us
#   What's a contour? Basically a list of points that make up a shape
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# The hope being the outline of the card should be the largest contour/shape
#   There will be smaller contours/shapes, in the art and what not, but they should be smaller than the card!
largest_contour = max(contours, key=cv2.contourArea)

epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# We might have something that isn't a perfect rectangle
# So we'll try to pick points along our shape and pick the 4 that make the largest area
# This should be the 4 corners of the card
max_area = 0
best_quad = None
# We'll try every combination of 4 points and pick the one with the largest area
for quad in combinations(approx, 4):
    quad = np.array(quad, dtype="float32")
    area = cv2.contourArea(quad)
    if area > max_area:
        max_area = area
        best_quad = quad

points = np.squeeze(best_quad) #Best quad is a list of a list of points, we just want a list of points

#Sort the points so their in clockwise order ie
# . Origin
#
#   0 1
#   3 2
# 
# v+Y   >+X
rect = np.zeros((4, 2), dtype="float32")
s = points.sum(axis=1) #Add together each points X and Y
rect[0] = points[np.argmin(s)] #Top left will have the smallest X, and smallest Y, and as such the smallest sum
rect[2] = points[np.argmax(s)] #Bottom right will have the largest X, and largest Y, and as such the largest sum
diff = np.diff(points, axis=1) #Subtract the X from the Y
rect[1] = points[np.argmin(diff)] #Top right will have the largest X and largest Y, so the smallest difference
rect[3] = points[np.argmax(diff)] #Bottom left will have the smallest X and smallest Y, so the largest difference


top_width =  rect[1][0] - rect[0][0] 
bottom_width = rect[3][0] - rect[2][0]

left_height = rect[0][1] - rect[3][1]
right_height = rect[2][1] - rect[1][1]

width = int(max(top_width, bottom_width))
height = int(max(left_height, right_height))

# Define the destination points
dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

# Perform the perspective transformation
matrix = cv2.getPerspectiveTransform(rect, dst)
card_image = cv2.warpPerspective(aruco_cropped_image, matrix, (width, height))

cv2.imshow("card_image", card_image)
cv2.imwrite("card_image.jpg", card_image)
cv2.waitKey(0)
