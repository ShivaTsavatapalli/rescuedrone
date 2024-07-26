import cv2
from matplotlib import pyplot as plt

# Load class names
classNames = []
classFile = "Object_Detection_Files\\coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load model configuration and weights
configPath = "Object_Detection_Files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Object_Detection_Files\\frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

def displayImage(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imgRGB)
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # List of objects to detect
    objects_to_detect = []  # Add more objects if needed

    plt.ion()  # Turn on interactive mode
    fig = plt.figure()

    while True:
        success, img = cap.read()
        if not success:
            break
        result, objectInfo = getObjects(img, 0.45, 0.2, objects=objects_to_detect)
        
        # Print detected objects
        for obj in objectInfo:
            print(f"Detected: {obj[1]} at {obj[0]}")
        
        displayImage(result)
        
        # Check for 'q' key press to quit
        if plt.waitforbuttonpress(timeout=0.01) and plt.get_fignums():
            break

    cap.release()
    plt.close()
