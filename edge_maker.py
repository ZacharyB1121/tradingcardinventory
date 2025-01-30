import cv2
from itertools import combinations
import numpy as np


aruco_cropped_image = cv2.imread("aruco_image.jpg")

# Load the image
gray = cv2.cvtColor(aruco_cropped_image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 130, 255, 0) #130 is the threshold value, 0-255, I found it via experimentation
blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

#OpenCV can find the edges of what is hopefully our card
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Generate random colors for each contour
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in contours]

# Draw each contour with a different color
e2 = aruco_cropped_image.copy()
for i, contour in enumerate(contours):
    cv2.drawContours(e2, [contour], -1, colors[i], 3)
cv2.imwrite("contours.jpg", e2)

# The hope being the outline of the card should be the largest contour/shape
#   There will be smaller contours/shapes, in the art and what not, but they should be smaller than the card!
largest_contour = max(contours, key=cv2.contourArea)
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

e3 = cv2.drawContours(aruco_cropped_image.copy(), [approx], -1, (255, 0, 0), 3)
cv2.imwrite("approx_polygon.jpg", e3)

e4 = cv2.drawContours(aruco_cropped_image.copy(), [largest_contour], -1, (0, 255, 0), 3)
cv2.imwrite("largest_contour.jpg", e4)

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



gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 130, 255, 0) #130 is the threshold value, 0-255, I found it via experimentation
blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

#OpenCV can find the edges of what is hopefully our card
edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Generate random colors for each contour
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in contours]

# Draw each contour with a different color
e2 = card_image.copy()
for i, contour in enumerate(contours):
    cv2.drawContours(e2, [contour], -1, colors[i], 3)
cv2.imwrite("contours.jpg", e2)



cv2.imshow("card_image", e2)





cv2.imwrite("card_image.jpg", card_image)
cv2.waitKey(0)
