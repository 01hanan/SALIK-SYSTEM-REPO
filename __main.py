import random
import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
import math
import requests
from supervision.draw.color import Color
from ultralytics import YOLO
import supervision as sv
import argparse
import base64
from tracker import Tracker


def send_sms(cam, image_file, className, conf):

    """
    send_sms -> This method is responsible for alerting the municipality about the detected hazard via 
    an SMS message.
    The message includes the camera location, the detected hazard image, the hazard class/type, 
    and the confidence score of the detection process.

    """
    # 1) Since the 'Gateway' SMS API does not support image files, we first need to convert 
    # the image file to a URL using imgBB API.

    # Our API key from imgBB
    api_key = '023a2a1fd5cadc3a4dc7f73b891268d6'

    # Read the image file and encode it in 'base64'
    # Open the image file in binary mode, encode it in base64, and decode it as UTF-8
    # This prepares the image data for safe transmission as text-based data,
    # ensuring compatibility with various systems and protocols.
    
    with open(image_file, 'rb') as file:
        image_data = base64.b64encode(file.read()).decode('utf-8')

    # Define the API endpoint URL
    url = "https://api.imgbb.com/1/upload"

    # Define the payload data
    payload = {
        "key": api_key,
        "image": image_data
    }

    # Make a POST request to upload the image on imgBB
    response = requests.post(url, data=payload)

    # Parse the JSON response
    data = response.json()

    # Check if the request was successful
    if response.status_code == 200 and data['success']:

        # Get the URL of the hosted image
        image_url = data['data']['url']
        print(f"Image uploaded successfully. URL: {image_url}")

    else:
        print("Failed to upload the image.")
    

    # 2) Now, we need to construct the SMS message and send it to the concerned party.

    # The URL for SMS gateway
    sms_gateway_url = "http://REST.GATEWAY.SA/api/SendSMS"
    
    # Define the parameters for the SMS request
    params = {
        "api_id": "API71789973116",
        "api_password": "salik2023CS",
        "sms_type": "T",
        "encoding": "T",
        "sender_id": "Gateway.sa",
        "phonenumber": "966558688926",
        "textmessage": f"A {className} has been detected at {cam}\nWith accuracy {conf}.\nImage File URL {image_url}"
    }
    
    # POST request to send the SMS
    response = requests.post(sms_gateway_url, params=params)




def main():
    # Initialize YOLO model for pothole detection
    model = YOLO("best-Yolo8l.pt")
    model.fuse()

    # Initialize object tracker
    tracker = Tracker()
    colors = [(0, 0, 255) for _ in range(10)]  # Red bounding boxes

    

    classNames = ["Pothole"]

    cap = cv2.VideoCapture(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device:", device)
    detection_threshold = 0.5


    last_sms_time = {}
    COOLDOWN_PERIOD = 30

    while True:
        ret, frame = cap.read()
        results = model.predict(frame)

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                score = math.ceil(score * 100) / 100 
                labels = [
                f"{classNames[class_id]} {score}"
            ]

                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

            # Update the object tracker
            tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                # Draw red bounding boxes for potholes
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id % len(colors)], 2)

                # Get class name and confidence
                class_name = classNames[class_id]
                confidence = score

                # Create the text to display
                text = f"{class_name}: {confidence:.2f}"

                # # Calculate the position to put the text
                text_x = int(x1)
                text_y = int(y1)  # Position the text above the bounding box

                # # Put the text on the frame
                
                background_color = (0, 0, 255)  # BGR color code (red in this example)
                text_color = (255, 255, 255)  # BGR color code (white in this example)

                # Create a rectangle with the desired background color
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_w, text_h = text_size
                cv2.rectangle(
                    frame,
                    (text_x, text_y - text_h),  # Upper-left corner of the rectangle
                    (text_x + text_w, text_y),  # Lower-right corner of the rectangle
                    background_color,  # Background color
                    thickness=cv2.FILLED,  # Fill the rectangle
                )

                # Put the text on the frame with a transparent background
                cv2.putText(
                    frame,
                    text,
                    org=(text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5,
                    color=text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    bottomLeftOrigin=False
                )


                # Check if a pothole is detected and send SMS
                if score >= 0.70 and not track.sms_sent:
                    current_time = time.time()
                    last_sms_send_time = last_sms_time.get(track_id, 0)
                    if current_time - last_sms_send_time >= COOLDOWN_PERIOD:
                        # Check if the cooldown period has passed
                        track.sms_sent = True  # Mark the track as having sent an SMS
                        last_sms_time[track_id] = current_time  # Update the last SMS send time
                        image_filename = "pothole_image.jpg"
                        cv2.imwrite(image_filename, frame)  # Save the image to a file
                        send_sms("#Makkah Region", image_filename, class_name, confidence)
                                        

            cv2.imshow("YOLOv8 Detection", frame)

        if ( cv2.waitKey(30) == 27 ):
                break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
            




if __name__ == "__main__":
    main()
    

