import cv2
import numpy as np

name = "vbunny"
name1 = './teaser/' + name + '/mesh.png'
name2 = './teaser/' + name + '/ours.png'
name3 = './teaser/' + name + '/vicini.png'

# Open the images
img1 = cv2.imread(name3)
img2 = cv2.imread(name2)
img3 = cv2.imread(name1)

# Check if images opened successfully
if img1 is None or img2 is None or img3 is None:
    print("Error opening image")
    exit()

# Get the properties of the images
frame_width = max(img1.shape[1], img2.shape[1], img3.shape[1])
frame_height = max(img1.shape[0], img2.shape[0], img3.shape[0])

# Resize the images to have the same size
if img1.shape != img2.shape or img1.shape != img3.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img3 = cv2.resize(img3, (img1.shape[1], img1.shape[0]))

# # Create masks for the split lines
mask1 = np.zeros((frame_height, frame_width), dtype=np.uint8)
mask2 = np.zeros((frame_height, frame_width), dtype=np.uint8)
mask3 = np.zeros((frame_height, frame_width), dtype=np.uint8)

# cv2.rectangle(mask1, (0, 0), (frame_width, frame_height//3), (255), thickness=-1)
# cv2.rectangle(mask2, (0, frame_height//3), (frame_width, 2*frame_height//3), (255), thickness=-1)
# cv2.rectangle(mask3, (0, 2*frame_height//3), (frame_width, frame_height), (255), thickness=-1)

# # lucy
# c = 60
# d = 80

c = 0
d = 0

# Define points for each mask (vbunny)
points1 = np.array([[[0, 0], [frame_width, 0], [frame_width, frame_height//4-c], [0, frame_height//2-c]]]) 
points2 = np.array([[[0, frame_height//2+1-c], [frame_width, frame_height//4+1-c], [frame_width, frame_height//2-d], [0, 3*frame_height//4-d]]])  
points3 = np.array([[[0, 3*frame_height//4+1-d], [frame_width, frame_height//2+1-d], [frame_width, frame_height], [0, frame_height]]])

# Fill each mask
cv2.fillPoly(mask1, points1, 255)
cv2.fillPoly(mask2, points2, 255)
cv2.fillPoly(mask3, points3, 255)

# Apply the masks
img1 = cv2.bitwise_and(img1, img1, mask=mask1)
img2 = cv2.bitwise_and(img2, img2, mask=mask2)
img3 = cv2.bitwise_and(img3, img3, mask=mask3)

# Combine the images
img = cv2.add(cv2.add(img1, img2), img3)

# # Create a colored line on the mask (long lines)
# line_thickness = 2
# cv2.line(img, (0, frame_height//2), (frame_width, frame_height//4), (255, 255, 255), thickness=line_thickness)
# cv2.line(img, (0, 3*frame_height//4), (frame_width, frame_height//2), (255, 255, 255), thickness=line_thickness)

# # Create a colored line on the mask (short lines)
# line_thickness = 2
# cv2.line(img, (frame_width//3, frame_height*7//16), (frame_width*2//3, frame_height*5//16), (255, 255, 255), thickness=line_thickness)
# cv2.line(img, (frame_width//3, frame_height*9//16), (frame_width*2//3, frame_height*7//16), (255, 255, 255), thickness=line_thickness)

# # Create a colored line on the mask (mid lines)
# line_thickness = 2
# cv2.line(img, 
#     (int(0.75 * 0 + 0.25 * frame_width), int(0.75 * (frame_height//2-c) + 0.25 * (frame_height//4-c))),
#     (int(0.25 * 0 + 0.75 * frame_width), int(0.25 * (frame_height//2-c) + 0.75 * (frame_height//4-c))), 
#     (255, 255, 255), thickness=line_thickness
# )
# cv2.line(img,
#     (int(0.75 * 0 + 0.25 * frame_width), int(0.75 * (frame_height*3//4-d) + 0.25 * (frame_height//2-d))),
#     (int(0.25 * 0 + 0.75 * frame_width), int(0.25 * (frame_height*3//4-d) + 0.75 * (frame_height//2-d))), 
#     (255, 255, 255), thickness=line_thickness
# )

# Create a colored line on the mask
line_thickness = 2
# a = 0.75 # lucy
a = 0.8 # vbunny
# a = 0.75 # buddha
cv2.line(img,
    (int(a * 0 + (1-a) * frame_width), int(a * (frame_height//2-c) + (1-a) * (frame_height//4-c))),
    (int((1-a) * 0 + a * frame_width), int((1-a) * (frame_height//2-c) + a * (frame_height//4-c))), 
    (255, 255, 255), thickness=line_thickness
)
cv2.line(img,
    (int(a * 0 + (1-a) * frame_width), int(a * (frame_height*3//4-d) + (1-a) * (frame_height//2-d))),
    (int((1-a) * 0 + a * frame_width), int((1-a) * (frame_height*3//4-d) + a * (frame_height//2-d))), 
    (255, 255, 255), thickness=line_thickness
)

# Save the image
cv2.imwrite('output.png', img)