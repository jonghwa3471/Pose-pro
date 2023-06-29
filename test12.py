from flask import Flask, Response, render_template
import sys
sys.path.append(r'C:\Users\USER\AppData\Local\Programs\Python\Python311\Lib\site-packages')
import cv2

app = Flask(__name__)

# OpenPose 코드
width = 320
height = 240
frame_skip = 2
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_iter_160000.caffemodel"

# Load the network
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def gen():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame_count = 0
    points = []
    threshold = 0.1
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while True:
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Read frame from the camera
        ret, frame = cap.read()

        # Apply OpenPose on the frame
        # Add your code here...
        # frame.shape = Get the frame's height, width, color channels
        imageHeight, imageWidth, _ = frame.shape

        # Preprocess for the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

        # Set the prepared blob as input to the network
        net.setInput(inpBlob)

        # Get the result
        output = net.forward()

        # The output shape[0] = image ID, [1] = height of the output map, [2] = width
        H = output.shape[2]
        W = output.shape[3]

        # Go through all the body parts
        for i in range(len(BODY_PARTS)):
            # For each body part, we get a heat map
            probMap = output[0, i, :, :]

            # Find the global maxima
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale to frame size
            x = (frame_width * point[0]) / W
            y = (frame_height * point[1]) / H

            if prob > threshold:
                # If the probability is more than the threshold, we add the point to the list of points
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[BODY_PARTS[partA]] and points[BODY_PARTS[partB]]:
                cv2.line(frame, points[BODY_PARTS[partA]], points[BODY_PARTS[partB]], (0, 255, 255), 2)
                cv2.circle(frame, points[BODY_PARTS[partA]], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
