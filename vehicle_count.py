# Import necessary packages
import cv2
import csv
import collections
import numpy as np
from tracker import * # import of all functions, classes and variables defined in the "tracker" module



# Initialize Tracker
tracker = EuclideanDistTracker() 
# Instance of cv2.VideoCapture class used to capture video from a file or camera
cap = cv2.VideoCapture('1.mp4')
input_size = 320



# Detection confidence thresholds
confThreshold = 0.2 # Confidence threshold is used to set a minimum level of confidence that an object has been detected correctly. Here, any object detection with a
#confidence score lower than 0.2 will be ignored. Only if there is more than 20 percent chance that the object is present then it will be considered as detection
nmsThreshold = 0.2 #technique used to filter out multiple detections of the same object. If "nmsThreshold" is set to 0.2 of the total area of either bounding box, then 
# the algorithm will consider them as a single detection and discard the one with lower confidence score



font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 1



# Middle cross line position
middle_line_position = 225
up_line_position = middle_line_position + 15
down_line_position = middle_line_position - 15



# Store Coco Names in a list
classesFile = "C:\Backup\Backup\Coding\Projects\Vehicle Counting And Classification\Vehicle-Counting-and-Classification\coco.NAMES"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))



# class index for our required detection classes
required_class_index = [2, 3, 5, 7] # indexing of python list starts from zero and initializing the variable to the index of car, motorbike, bus and truck
detected_classNames = []# an empty python list 



# Model Files
modelConfiguration = 'yolov3-320.cfg' # configuration file contains the network architecture and other hyperparameters of the model
modelWeigheights = 'yolov3-320.weights' # file path to the pretrained weights file
# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights) # net is an instance of "cv2.dnn" module in the OpenCV library to read a neural network model 
#trained using the darknet framework of the YOLOv3 (You Only Look Once) is by default trained on the coco dataset. The original YOLOv3 paper references the COCO dataset.
# COCO dataset is a large scale image recognition dataset
# designed for object detection, segmentation, and captioning. It contains over 3,30,000 images, 1.5 million object instances and 80 object classes. The two arguments are the paths to the files containing the model configuration and model weights respectively
#Yolo is a real time object detection system and is a single convolutional neural network that can detect objects in images and videos with a high degree of accuracy and speed. It was
# first introduced by Joseph Redmon and others in their 2016 research paper. Here we are using yolo version 3 published in 2018. Various software libraries or platforms have been developed 
# to implement the YOLO object detection system and are known as YOLO frameworks. These frameworks provide and environment for user to train and deploy YOLO models for object detection tasks
#eg. Darknet, Tensorflow, Pytorch, opencv, Keras. The original implementation of YOLO was done in the Darknet framework, which is an open source neural network framework written in C and CUDA
# It provdes a simple and efficient environment for training anf deploying YOLO models



# Configure the network backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #sets the DNN backend to CUDA. DNN backend refers to the library or framework that is used to perform the computations. 
# Opencvs DNN module supports several backends, including CUDA, OpenCL, and CPU. CUDA is a parallel computing platform and API for using a GPU to accelerate computationally intensive tasks.
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)# sets the DNN target to CUDA, which means that the computations for the DNN will be executed on the CUDA-enabled GPU. In summary,
#DNN backend refers to the library or framework that the DNN is using to perform the computations and DNN target is the specific device or hardware the computations are being 
#executed on. The first line sets the DNN backend to CUDA, which means that it will use the CUDA library to perform the computations. The second line is setting the DNN target to 
# CUDA as well, which means that the computations will be executed on a CUDA enabled GPU



# Define random colour for each class
np.random.seed(42)#sets the seed for the pseudorandom number generator in the numpy library to 42 meaning that any subsequent calls to function from numpy's random module s/a "np.random.rand"
# or "np.random.randn()", will produce the same sequence of numbers as if the seed were set to 42.
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')# generates an array of random integers b/w 0 and 255(inclusive) of the size of "length of classnames list *3". The 'uint8'
#dtype means that the integers will be stored as 8-bit unsigned integers. The resulting array will be used to store RGB color values for each class
print(len(colors))


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy



# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]



# Function for count vehicle
def count_vehicle(box_id, img): # counting vehicles relative to their position in three lines
    x, y, w, h, id, index = box_id # box id is a tuple. Tuple is immutable ordered collection of elements declared using paranthesis. Unlike list its elements cannot be modified
    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    # Find the current position of the vehicle
    if (iy > down_line_position) and (iy < middle_line_position):
        if id not in temp_down_list:
            temp_down_list.append(id)
    elif iy < up_line_position and iy > middle_line_position:
        if id not in temp_up_list:
            temp_up_list.append(id)
    elif iy < down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index]+1
    elif iy > up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1
    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)



# Function for finding the detected objects from the network output
def postProcess(outputs, img):
    global detected_classNames # variable could have been acessed without redeclaratrion but global redeclaration allows modification and modifications made in the code will also be reflected in the global scope
    height, width = img.shape[:2] # img.shape is a property of a numpy array representing an image, which returns a tuple containing the number of rows, columns and channels. The code returns tuple containing 
    #number of rows and columns of the image, which are equivalent to the height and width of the image respectively
    #img.shape[0] corresponds to the number of pixels in the vertical direction and img.shape[1] corresponds to the number of pixels in horizontal direction.
    boxes = []
    classIds = []
    confidence_scores = [] # represents the level of cofidence in prediction
    detection = []
    for output in outputs:# output is a list of detections(bounding boxes with class labels and confidence scores). the "ouputs" is a list of lists where each inner list contains one detection with the format
        # center_x, center_y, width, height, class1_score, class2_score, class3_score, ...
        for det in output:# The inner loop is iterating through each detection and extracts the scores(class probabilities) by slicing the detection array from the 5th index until the end.
            scores = det[5:] # This creates a sublist
            classId = np.argmax(scores) #numpy function argmax returns the index of the maximum value in the "scores" list.
            confidence = scores[classId] # stores the value of confidence score
            if classId in required_class_index: # If the index of highest confidence score matches with the index value of car, motorbike, bus or truck
                if confidence > confThreshold: # and if the value of the confidence score is greater than the threshold value
                    # print(classId)
                    w, h = int(det[2]*width), int(det[3]*height) # width and height represent the width and height of the resized frame that was passed and det[2] and det[3] represent the width and height of bounding 
                    # box of the object relative to input image in the range[0,1](calculated by dividing original coordinates and dimensions by pixel values).eg. if frame has resolution 512*512 pixels and bounding box
                    #  has coordinates(0.3,0.4,0.2,0.1) then 0.2*512 ans 0.1*512 give 102 and 51 approx hence bounding box has width of 102 pixels and height of 51 pixels.
                    x, y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)# x and y represent the top left corner of the bounding box in pixels
                    boxes.append([x, y, w, h]) # coordinates of all the boxes detected
                    classIds.append(classId) # append the index of the highest confidence score
                    confidence_scores.append(float(confidence)) # append that highest confidence score 
#width and height represent the width and height of the frame and det[2] and det[3] represent the width and height of detected image
    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold) #reduces the number of bounding boxes in an image by suppressing "overlapping" boxes that have a lower confidence score and returns a list 
    # of indices of the bounding boxes that were not suppressed
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%',
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append(
                [x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def realTime():
    while(cap.isOpened()): # cap an object of cv2.VideoCapture class used to capture video from a camera or a file. The "isOpened" method returns a Boolean value
        # indicating whether the video capture object has been successfully opened or not. As long as the video capture object is open, the loop continues to execute inside it
        success, img = cap.read() #Function is used to read the next frame from a video file or a camera connected to the computer and return two tuples. "success" a boolean value
        # indicating if the frame was successfully read or not. "img" holds image data, a multidimensional array of numbers where each number represents the color or intensity of a
        # corresponding pixel in the image
        if success:
            img = cv2.resize(img, (0, 0), None, 0.5, 0.5) # to resize the read frame.second argument specifies target size of image after resizing. In this case the tuple is set to (0,0)
            # which means that the target size is not specified and the image will be resized according to the next two arguments. Third argument is used to specify the interpolation method,
            # its set to None, so it defaults to cv2.INTER_LINEAR which is bilinear interpolation. Interpolation is the process of estimating the pixel values in a new image based on the pixel
            # values of an original image. Also the width and height of the image will be scaled down by a factor of 0.5
            ih, iw, channels = img.shape # height,width and number of channels(r,g,b) of the image represented by img variable. colour image has three whereas grayscale has only one channel
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False) # fn returns a 4 dimensional binary large object that can be passed as an input to a deep
            # learning model with shape (1,3,input_size,input_size) where 1 represents batch size or number of images,3 is the number of color channels per image, twice occurence of input_size 
            # represents the height and width of each image respectively. The scaling factor 1/255 is used to normalize the pixel values to a range between 0 and 1, a common preprocessing for deep
            # learning models. The pixel values in the image range from 0 to 255, where 0 is black and 255 is white. A pixel is the smallest unit of a digital image. The third argument specifies the
            # mean value that should be subtracted from each channel of the image. 1 is a flag value specifing that the channel ordering is not swapped. False specifies that image should not be cropped after resizing 
            # Set the input of the network
            net.setInput(blob) # sets the blob as an input to the neural network represented by net object trained using darknet framework of yolov3
            layersNames = net.getLayerNames() # returns a list of strings representing the names of layers in the network ordered from the input to the output of the network
            outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()] # creates a list that contains the names of layers not connected to other layers (o/p layers) in the network "net".The
            # function returns a list of indices of such layers and the list comprehension iterates through the list using the index to access the corresponding name from the "layersNames" list. The index difference
            # is because the indexing of layers in the network starts at 0, while indexing of elements in a list starts at 1
            # Feed data to the network
            outputs = net.forward(outputNames)# runs forward propagation on the network and returns the output from the output layers specified in "outputNames", basically a list of numpy arrays, where each array corresponds
            # to the output from one of the o/p layers. Each o/p array contains information about objects detected in the image s/a their bounding boxes, class labels, and confidence scores. The specific classes that YOLOv3 can 
            #detect will depend on the dataset used to train the model. Original YOLOv3 paper used the COCO dataset, which contains 80 object classes
            #print(len(outputNames)) 
            postProcess(outputs, img) # Find the objects from the network output # function call

            # Draw the crossing lines

            cv2.line(img, (0, middle_line_position),
                     (iw, middle_line_position), (0, 255, 0), 2)
            cv2.line(img, (0, up_line_position),
                     (iw, up_line_position), (255, 0, 0), 2)
            cv2.line(img, (0, down_line_position),
                     (iw, down_line_position), (0, 0, 255), 2)

            # Draw counting texts in the frame
            cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, font_color, font_thickness)
            cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, font_color, font_thickness)
            cv2.putText(img, "Car:        "+str(up_list[0])+"     " + str(
                down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     " + str(
                down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Bus:        "+str(up_list[2])+"     " + str(
                down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Truck:      "+str(up_list[3])+"     " + str(
                down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

            # Show the frames
            cv2.imshow('Output', img)

            if cv2.waitKey(1) == ord('q'):
                break

        else:
            break
    # Write the vehicle counting information in a file and save it

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


image_file = 'img.jpg'


def from_static_image(image):
    img = cv2.imread(image)

    blob = cv2.dnn.blobFromImage(
        img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(outputNames)
    # Find the objects from the network output
    postProcess(outputs, img)

    # count the frequency of detected classes
    frequency = collections.Counter(detected_classNames)
    print(frequency)
    # Draw counting texts in the frame
    cv2.putText(img, "Car:        "+str(frequency['car']), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorbike:  "+str(frequency['motorbike']), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus:        "+str(frequency['bus']), (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck:      "+str(frequency['truck']), (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

    cv2.imshow("image", img)

    cv2.waitKey(0)

    # save the data to a csv file
    with open("static-data.csv", 'a') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow([image, frequency['car'], frequency['motorbike'],
                         frequency['bus'], frequency['truck']])
    f1.close()


if __name__ == '__main__': #entry point of the script
    realTime()
    # from_static_image(image_file)













