
![fos-low-resolution-color-logo](https://github.com/NiranjanNR/FOS/assets/84635960/c35645f2-4f38-47b8-b077-ce42de66da05)

## FOS is an attempt at contributing to the increase in energy efficient technology using the rising power of artificial intelligence and deep learning.

The whole idea of FOS can be reflected through using OpenCV with deep learning models like TFLite and mediapipe to understand what activity 
a person is indluged in and to control the luminance of a bulb used to light the given environment. This can not only prove really helpful 
to the person using the bulb and save him efforts in adjusting the light himself but it can also prove really helpful by cutting the excess
energy that would rather be spent on unnecessary extra lighting. 

This model can predict activities like using a phone, laptop or even reading a book and adjust the lighting accordingly. For example using a
phone requires low light, while reading required brighter light, so it adjusts the lighting accordingly.

The whole project was done on a Raspberry Pi 4 as we required a high performance Iot device to control the lights. We integrated the use of
the above specified deep learning models for object detection and pose detection to understand what the person is doing. Then we used the 
library called gpiozero to use the pins of the raspberry pi 4 to control LEDs to show for the proof of concept.

The following gif can show our proof of concept.

https://github.com/NiranjanNR/FOS/assets/84635960/cd15f741-ab64-4758-9ba1-602a6f16ae61

After setting up a webcam in the required position one must follow these steps to run this project on a Raspberry Pi 4:
First create a folder that you want this app to run in. Then install dependencies like OpenCv and Mediapipe to the virtual environment or globally. Into the folder that you created, clone the repository of TFLite object detection from GitHub (Link:https://github.com/tensorflow/examples). Inside the cloned folder locate lite/examples/object_detection/raspberry_pi/detect.py file following the path specified. After finding this file replace it with the detect.py file uploaded. Then the user must run this python file in the current environment to execute the project. This execution will recognize what the person is doing along with prompting the lighting required for the room. In order to see this project in action, the Raspberry Pi 4 must be connected to 3 LEDs. This connection must done to the GPIO pins specified in the detect.py file. Once that is done the program will trigger the right LEDs for each action performed by the user.
