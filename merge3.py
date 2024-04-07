
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

# Open the images
img1 = cv2.imread('/home/zw336/IR/differentiable-sdf-rendering/exp/vbunny_mesh/final/frames/frame21.png')
img2 = cv2.imread('/home/zw336/IR/differentiable-sdf-rendering/exp/vbunny_env_fixed3 (best)/final/frames/frame21.png')
img3 = cv2.imread('/home/zw336/IR/differentiable-sdf-rendering/exp/vbunny_env_fixed3 (best)/final/frames/frame21.png')

# Check if images opened successfully
if img1 is None or img2 is None:
    print("Error opening image")
    exit()

# Get the properties of the images
frame_width = max(img1.shape[1], img2.shape[1])
frame_height = max(img1.shape[0], img2.shape[0])
print("frame_width: ", frame_width)
print("frame_height: ", frame_height)

# Resize the images to have the same size
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# # Create a 3/4 mask for the split line
mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
# points = np.array([[[frame_width*3//4, 0], [frame_width//4, frame_height], [frame_width, frame_height], [frame_width, 0]]])
# points = np.array([[[frame_width*5//8, 0], [frame_width*3//8, frame_height], [frame_width, frame_height], [frame_width, 0]]])
# points = np.array([[[frame_width*9//16-50, 0], [frame_width*7//16-50, frame_height], [frame_width, frame_height], [frame_width, 0]]])
points = np.array([[
    [frame_width*9//16, 0], 
    [frame_width*7//16, frame_height], 
    [frame_width, frame_height], 
    [frame_width, 0]
]])
cv2.fillPoly(mask, points, 255)

# # Create a 1/2 mask for the split line
# mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
# points = np.array([[[frame_width//2, 0], [frame_width//2, frame_height], [frame_width, frame_height], [frame_width, 0]]])
# cv2.fillPoly(mask, points, 255)

# Split the images
img1 = cv2.bitwise_and(img1, img1, mask=cv2.bitwise_not(mask))
img2 = cv2.bitwise_and(img2, img2, mask=mask)

# Combine the images
img = cv2.add(img1, img2)

# Define the font
font = cv2.FONT_HERSHEY_TRIPLEX

# Add the text
# cv2.putText(img, 'Ours', (10,50), font, 2, (255,255,255), 2, cv2.LINE_AA)
# cv2.putText(img, 'GT', (frame_width-100,frame_height-10), font, 2, (255,255,255), 2, cv2.LINE_AA)

# Create a colored line on the mask
line_thickness = 2
# cv2.line(img, (frame_width//2, 0), (frame_width//2, frame_height), (255, 255, 255), thickness=line_thickness)
# cv2.line(img, (frame_width*3//4, 0), (frame_width//4, frame_height), (255, 255, 255), thickness=line_thickness)
# cv2.line(img, (frame_width*5//8, 0), (frame_width*3//8, frame_height), (255, 255, 255), thickness=line_thickness)
# cv2.line(img, (frame_width*9//16-50, 0), (frame_width*7//16-50, frame_height), (255, 255, 255), thickness=line_thickness)
cv2.line(img, (frame_width*9//16, 0), (frame_width*7//16, frame_height), (255, 255, 255), thickness=line_thickness)

# Save the image
cv2.imwrite('output.png', img)

# Convert the image from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a new PDF file
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

# Create a new figure
fig = plt.figure(dpi=300)

# Add the image to the figure
plt.imshow(img)
plt.axis('off')  # to hide the axis

# Save the figure to the PDF file
pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

# Close the PDF file
pdf.close()