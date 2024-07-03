from flask import Flask, flash, request, redirect, url_for, render_template, Response,jsonify
from flask_cors import CORS, cross_origin
import json
import time
import os
import cv2
import draw
from io import BytesIO
import base64
from PIL import Image
import re
from pixel_image import atkinson_dither
import numpy as np
import time
from werkzeug.utils import secure_filename
import math
from flask import send_file


from flask import send_file
import io 
min_dist = 50
app = Flask(__name__)
CORS(app)
############################
app.secret_key = "Social distancing"
UPLOAD_FOLDER = 'static/video_upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#############################social distancing and camera parameters############################################
CONFIDENCE_THRESHOLD = 0
NMS_THRESHOLD = 0.4
net = cv2.dnn.readNet('model/yolov4.weights', 'model/yolov4.cfg')
model = cv2.dnn_DetectionModel(net)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


######################################Home################################################################################
@app.route('/home')
@cross_origin()
def home():
    return render_template('index_social_distancing.html')


###################################WEBCAM###################################################
# @app.route('/webcam')
# def index():
#     return render_template('layout.html')

# @app.route('/process', methods=['POST'])
# def process():
#     input = request.json
#     image_data = re.sub('^data:image/.+;base64,', '', input['img'])
#     image_ascii = atkinson_dither(image_data)
#     # print(image_ascii)
#     return image_ascii
###################################WEBCAM-END###################################################

def detector(frame):
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return classes, scores, boxes


#############################################IMAGE TEST##############################################
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'jfif'])
def allowed_file(filename1):
    # print("orinhht1")
    dot_index = filename1.index('.')
    ext = filename1[(dot_index + 1):].lower()
    # ext=filename1[-3:].lower()
    # print(ext)
    if (ext in ALLOWED_EXTENSIONS):
        return True
    else:
        return False


@app.route("/image_upl")
def image_home():
    return render_template('image_upload_html.html')


def PIL_to_base64(pil_image):
    buffered = BytesIO()
    img1 = pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())


@app.route('/image_upload', methods=['GET', 'POST'])
@cross_origin()
def upload_image():
    min_dist = 100  # 100 for uploaded images, 300 for webcam
    if 'file1' not in request.files:
        flash('No files part')
        return 'No image selected for uploading'

    file1 = request.files['file1']
    if file1.filename == '':
        flash('No image selected for uploading')

    if allowed_file(file1.filename):
        filestr1 = file1.read()
        npimg1 = np.frombuffer(filestr1, np.uint8)
        frame = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
        
        classes, scores, boxes = detector(frame)
        draw.drawing(classes, scores, boxes, frame, min_dist)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode('.jpg', frame)
        image_data = io.BytesIO(encoded_image.tobytes())

        # Return the image data as a response
        return send_file(image_data, mimetype='image/jpeg')

    return 'Invalid file format'

@app.route('/display', methods=['post'])
def display_image(filename1, filename2):
    return render_template('image_upload_html.html', filename1=filename1, filename2=filename2)


#######################################IMAGE END ################################################
###################################VIDEO TEST##################################################################
ALLOWED_EXTENSIONS_VID = set(['mp4', 'avi', 'wmv'])

def allowed_file_vid(filename1):
    ext = filename1[-3:].lower()
    return ext in ALLOWED_EXTENSIONS_VID

@app.route("/video_upl")
@cross_origin()
def video_home():
    return render_template('video_upload_html.html')

@app.route("/video_upload", methods=['GET', 'POST'])
@cross_origin()
def upload_video():
    if 'file2' not in request.files:
        flash('No files part')
        return 'No video selected for uploading'

    file2 = request.files['file2']
    if file2.filename == '':
        flash('No video selected for uploading')
        return 'No video selected for uploading'

    if allowed_file_vid(file2.filename):
        filename1 = secure_filename(file2.filename)
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], "test_vid.mp4"))
    
    return gen_frame()

def dist(pt1, pt2):
    try:
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    except:
        return

def gen_frame():
    min_dist = 100  # 100 for uploaded images and video, 300 for webcam
    cap = cv2.VideoCapture("./static/video_upload/test_vid.mp4")
    dir = './static/uploads/vid_frames'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    
    (grabbed, frame) = cap.read()
    if not grabbed:
        exit()
    else:
        count = 0
        while grabbed:
            start = time.time()
            classes, scores, boxes = detector(frame)  # Assuming you have the detector function defined
            end = time.time()
            draw.drawing(classes, scores, boxes, frame, min_dist)  # Assuming you have the draw function defined
            path1 = f"./static/uploads/vid_frames/frame{count}.png"
            cv2.imwrite(path1, frame)
            grabbed, frame = cap.read()
            count += 1

    # Set up YOLOv4 model
    net = cv2.dnn.readNet('model/yolov4.weights', 'model/yolov4.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Set up video capture
    cap = cv2.VideoCapture("./static/video_upload/test_vid.mp4")
    _, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_path = 'output.avi'
    writer = cv2.VideoWriter(out_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # Main loop
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            height, width = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    if class_id != 0:
                        continue
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            persons = []
            person_centres = []
            violate = set()
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    persons.append(boxes[i])
                    person_centres.append([x + w // 2, y + h // 2])
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    if dist(person_centres[i], person_centres[j]) <= min_dist:
                        violate.add(tuple(persons[i]))
                        violate.add(tuple(persons[j]))
            v = 0
            for (x, y, w, h) in persons:
                if (x, y, w, h) in violate:
                    color = (0, 0, 255)
                    v += 1
                else:
                    color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.circle(img, (x + w // 2, y + h // 2), 2, (0, 0, 255), 2)
            cv2.putText(img, 'No of Violations : ' + str(v), (15, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 126, 255), 2)
            writer.write(img)
            cv2.imshow('Output', img)

            if cv2.waitKey(1) == 27:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return send_file(out_path,mimetype='video/mp4/avi')


#########################################video_feed for video#####################################
@app.route('/video_feed_video', methods=['GET','POST'])
def video_feed_video():
    return Response(
        video_gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



def calculate_distance(box1, box2):
    # Calculate the centroid of each bounding box
    center_x1 = box1[0] + box1[2] / 2
    center_y1 = box1[1] + box1[3] / 2
    center_x2 = box2[0] + box2[2] / 2
    center_y2 = box2[1] + box2[3] / 2
    
    # Calculate Euclidean distance
    distance = np.sqrt((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2)
    return distance
def process_detection(frame):
    # Perform object detection
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    # Initialize the list to store distances between people
    distances = []

    print("Detected boxes:", boxes)  # Debugging: Print the detected boxes

    # Calculate distance between people and draw distances on the image
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            distance, center1, center2 = calculate_distance(boxes[i], boxes[j])
            distances.append({
                'person1': i + 1,
                'person2': j + 1,
                'distance': distance
            })
            if distance < min_dist:
                # Violation detected, you can perform actions here
                print(f"Violation: Distance between person {i+1} and person {j+1} is {distance}")
            
            # Draw distance on the image
            print("Center points:", center1, center2)  # Debugging: Print center points
            cv2.line(frame, center1, center2, (255, 0, 0), 2)
            mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
            cv2.putText(frame, f'{distance:.2f}', mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw bounding boxes
    for i, box in enumerate(boxes):
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {i + 1}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode the image and create BytesIO object
    _, encoded_image = cv2.imencode('.jpg', frame)
    image_data = BytesIO(encoded_image.tobytes())
    
    # Return the distances and the image data
    return distances, image_data



@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        distances, image_data = process_detection(frame)

        # You can return the distances and image as a JSON response
        response = {
            'distances': distances,
            'image': base64.b64encode(image_data.getvalue()).decode('utf-8')
        }
        return jsonify(response)

    return 'Invalid file format', 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    # SERVER=Server(app.wsgi_app)
    # SERVER.serve()
    # app.config['TEMPLATES_AUTO_RELOAD'] = True
    # app.run(host="0.0.0.0",threaded=True,port=8010,debug=True)
    # app.run(host='0.0.0.0',threaded=True, port=5014,debug=True)
    app.run(debug=True)
