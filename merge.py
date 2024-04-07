import cv2
import numpy as np

# Open the videos
# cap1 = cv2.VideoCapture('/home/zw336/IR/differentiable-sdf-rendering/exp/buddha_env_fixed/final/turntable.mp4')
# cap2 = cv2.VideoCapture('/home/zw336/IR/differentiable-sdf-rendering/exp/buddha_mesh/final/turntable.mp4')

cap1 = cv2.VideoCapture('/home/zw336/IR/differentiable-sdf-rendering/exp/kangaroo/final/turntable.mp4')
cap2 = cv2.VideoCapture('/home/zw336/IR/differentiable-sdf-rendering/exp/kangaroo_mesh/final/turntable.mp4')

# Check if videos opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error opening video")
    exit()

# Get the properties of the videos
fps = cap1.get(cv2.CAP_PROP_FPS)
frame_width = int(max(cap1.get(cv2.CAP_PROP_FRAME_WIDTH), cap2.get(cv2.CAP_PROP_FRAME_WIDTH)))
frame_height = int(max(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT), cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Create a VideoWriter object
out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width, frame_height))

# Define the font
font = cv2.FONT_HERSHEY_TRIPLEX

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 == True and ret2 == True:
        # Resize the frames to have the same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # # Create a 3/4 mask for the split line
        # mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        # points = np.array([[[frame_width*3//4, 0], [frame_width//4, frame_height], [frame_width, frame_height], [frame_width, 0]]])
        # cv2.fillPoly(mask, points, 255)

        # # Create a 1/2 mask for the split line
        # mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        # points = np.array([[[frame_width//2, 0], [frame_width//2, frame_height], [frame_width, frame_height], [frame_width, 0]]])
        # cv2.fillPoly(mask, points, 255)

        # Create a 9/16 mask for the split line
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        points = np.array([[[frame_width*9//16, 0], [frame_width*7//16, frame_height], [frame_width, frame_height], [frame_width, 0]]])
        cv2.fillPoly(mask, points, 255)

        # Split the frames
        frame1 = cv2.bitwise_and(frame1, frame1, mask=cv2.bitwise_not(mask))
        frame2 = cv2.bitwise_and(frame2, frame2, mask=mask)

        # Combine the frames
        frame = cv2.add(frame1, frame2)

        # Add the text
        cv2.putText(frame, 'Ours', (10,50), font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'GT', (frame_width-100,frame_height-10), font, 2, (255,255,255), 2, cv2.LINE_AA)

        # Create a colored line on the mask
        line_thickness = 2
        # cv2.line(frame, (frame_width*3//4, 0), (frame_width//4, frame_height), (255, 255, 255), thickness=line_thickness)
        # cv2.line(frame, (frame_width//2, 0), (frame_width//2, frame_height), (255, 255, 255), thickness=line_thickness)
        cv2.line(frame, (frame_width*9//16, 0), (frame_width*7//16, frame_height), (255, 255, 255), thickness=line_thickness)

        # Write the frame into the file 'output.mp4'
        out.write(frame)
        print("frame written")
    else:
        break

# Release everything
cap1.release()
cap2.release()
out.release()