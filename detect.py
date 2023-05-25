# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from gpiozero import LED
from time import sleep
red1 = LED(17)
red2 = LED(27)
red3 = LED(23)


#IMPORTING MEDIAPIPE
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          sys.exit(
              'ERROR: Unable to read from webcam. Please verify your webcam settings.'
          )

        counter += 1
        image = cv2.flip(image, 1)
    
        # Convert the image from BGR to RGB as required by the Mediapipe model.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #MEDIAPIPE COMES IN:
        results = pose.process(rgb)
        left_index = []
        right_index = []
        left_wrist = []
        right_wrist = []
        try:
            landmarks = results.pose_landmarks.landmark
            #Landmarks
            left_index=[landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            right_index=[landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            left_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        except:
            pass
        
        
        
                
        #Drawing
        image.flags.writeable = True
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)
        
        #fudhi upayogikunna bhagam
        info = detection_result.detections
        for i in info:
            try:
                if i.categories[0].category_name == "person" :
                    red1.on()
                    red2.off()
                    red3.off()
            except:
                pass
            
            try:
                if i.categories[0].category_name == "cell phone" or i.categories[0].category_name == "remote" :
                    phone_in_hand_or_not_left=abs((right_index[0]+right_wrist[0])/2*1000-(i.bounding_box.origin_x+i.bounding_box.width//2)-50)
                    phone_in_hand_or_not_right=abs((left_index[0]+left_wrist[0])/2*1000-(i.bounding_box.origin_x+i.bounding_box.width//2)-250)
                    if phone_in_hand_or_not_right < 100 or phone_in_hand_or_not_left < 100:
                        cv2.putText(image, "Using Phone = Low Light",
                           (35,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA 
                           )
                        red1.off()
                        red2.on()
                        sleep(2)
            
            except:
                pass
            
            try:
                if i.categories[0].category_name == "book":
                    print(i.bounding_box.origin_x+i.bounding_box.width//2)
                    book_in_hand_or_not_right=abs((right_index[0]+right_wrist[0])/2*1000-(i.bounding_box.origin_x+i.bounding_box.width//2)-50)
                    book_in_hand_or_not_left=abs((left_index[0]+left_wrist[0])/2*1000-(i.bounding_box.origin_x+i.bounding_box.width//2)-250)
                    if book_in_hand_or_not_right < 250 or book_in_hand_or_not_left < 250:
                        cv2.putText(image, "Reading = Bright Light",
                           (35,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA 
                           )
            except:
                pass
            
            try:
                if i.categories[0].category_name == "laptop":
                    print(i.bounding_box.origin_x+i.bounding_box.width//2)
                    book_in_hand_or_not_right=abs((right_index[0]+right_wrist[0])/2*1000-(i.bounding_box.origin_x+i.bounding_box.width//2)-50)
                    book_in_hand_or_not_left=abs((left_index[0]+left_wrist[0])/2*1000-(i.bounding_box.origin_x+i.bounding_box.width//2)-250)
                    if book_in_hand_or_not_right < 250 or book_in_hand_or_not_left < 250:
                        cv2.putText(image, "Using Laptop = Low Light",
                           (35,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA 
                           )
                        red1.on()
                        red2.on()
                        red3.on()
                        sleep(2)
            except:
                pass
        #print(detection_result.detections[0].categories[0].category_name)

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result)
    

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
          end_time = time.time()
          fps = fps_avg_frame_count / (end_time - start_time)
          start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
          red1.off()
          red2.off()
          red3.off()
          break
        cv2.imshow('object_detector', image)

      cap.release()
      cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
