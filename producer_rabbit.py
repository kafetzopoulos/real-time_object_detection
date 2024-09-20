from ultralytics import YOLO
import base64
import pika
import json
import cv2

# initializing YOLO model (I am currently using the s version from the ROC curve, due to slow hardware)
model = YOLO("yolov5s.pt")
# model = YOLO("yolov8x.pt")

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

# the absolute path to the video
video_path = r"J:\Productivity\pythonProject\Object_Detection_in_Real_time\data\6896028_fhd_30fps.mp4"

# this function uses a single frame
# encodes it to a JPEG
# converts it to a JSON object
# and finally publishes it to the RabbitMQ queue (for the consumer to use it)
def callback(frame):

    # applying the trained model to the current frame, from either the .mp4, or webcam
    results = model(frame)

    # if the results aren't empty:
    if results:
        # I am interested only to the first element of the list
        result = results[0]

        # converting the detection results to JSON
        results_json = result.tojson()

        # encoding the frame to JPEG (I just need the buffer, thus using _ variable)
        _, buffer = cv2.imencode('.jpg', frame)

        # extracting the byte data from the buffer
        frame_bytes = buffer.tobytes()

        # encoding the JPEG bytes to base64 string so that it can be handled by the JSON object
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

        # the JSON format
        results_data = {
            'frame': frame_base64,  # the encoded JPEG
            'detections': json.loads(results_json)  # the detection results
        }

        # publishing to a RabbitMQ message queue
        channel.basic_publish(exchange='',
                              routing_key='object_detection_results',
                              body=json.dumps(results_data))
        print(" [✔] successfully sent to RabbitMQ")


# 2 options to choose from:
# use the .mp4 file
cap = cv2.VideoCapture(video_path)
#       or
# use the camera
# cap = cv2.VideoCapture(0)

while cap.isOpened():

    # reading the frame (ret is boolean)
    ret, frame = cap.read()

    # if ret is false, then the frame wasn't read successfully (exits)
    if not ret:
        break
    # if ret is True, it initiates the def
    callback(frame)

# releasing the video capture when the loop ends (frees web cam source)
cap.release()

# terminates connection to RabbitMQ
connection.close()
