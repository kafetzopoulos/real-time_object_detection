import base64
from flask import Flask, Response, render_template
import cv2
import numpy as np
import json
import pika

app = Flask(__name__)

# trying to connect to RabbitMQ
try:
    # establishing connection to localhost
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    print("Connected to RabbitMQ")
# throw exception if otherwise
except pika.exceptions.AMQPConnectionError as e:
    print(f"Connection error: {e}")
    exit()

# declaring the queue with name object_detection_results
channel.queue_declare(queue='object_detection_results')

# this function continuously receives messages from the queue
def generate_frames():

    print("📩 Received message")

    # infinite loop to keep checking for new messages in queue
    while True:
        # getting the message from the queue
        method_frame, header_frame, body = channel.basic_get(queue='object_detection_results', auto_ack=True)

        # if a message is available:
        if method_frame:

            # decoding the JSON message
            message = json.loads(body)

            # extracting the base64 frame
            frame_base64 = message['frame']

            # extracting the list of detections
            detections = message['detections']

            # trying to decode the frame
            try:
                # decoding the base64 string to bytes
                frame_bytes = base64.b64decode(frame_base64)

                # converting the bytes to a NumPy array
                np_arr = np.frombuffer(frame_bytes, np.uint8)

                # decoding the image from the NumPy array
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # if frame is empty: skip
                if frame is None:
                    continue

                # drawing the detection boxes around objects
                for detection in detections:
                    #  coordinates of each box in the frame (in order to overlay them)
                    x1, y1, x2, y2 = detection['box']['x1'], \
                                     detection['box']['y1'], \
                                     detection['box']['x2'], \
                                     detection['box']['y2']

                    # the confidence for each prediction and the name of each detected object
                    conf, cls = detection['confidence'], detection['name']

                    # drawing the boxes to the frame (green rectangle) (of thickness 2)
                    cv2.rectangle(frame,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)

                    # drawing the text of each rectangle
                    cv2.putText(frame,
                                f'{cls} {conf:.2f}',        # [name,confidence score]
                                (int(x1), int(y1) - 10),    # placement of text on rectangle
                                cv2.FONT_HERSHEY_SIMPLEX,   # font
                                0.9,                        # font scale
                                (0, 255, 0), 2)             # (same colour & thickness)

                # encoding the frame as a JPEG
                _, buffer = cv2.imencode('.jpg', frame)

                # converting it back to bytes
                frame = buffer.tobytes()

                # yielding the bytes in a suitable format for the HTTP
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # throwing exception if failing to decode the frame
            except Exception as e:
                print(f"Error processing frame: {e}")

# The main html page
@app.route('/')
def index():
    return render_template('index.html')

# the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  # continuously updating the stream

# running the Flask web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
