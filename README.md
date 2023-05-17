# FOS
![fos-transparent-bg](https://github.com/NiranjanNR/FOS/assets/84635960/91d9eed5-b879-4540-9313-1f7ae850ec99)

## FOS is an attempt at contibuting to the increase in energy efficient technology using the rising power of artificial intelligence and deep learning.

The whole idea of FOS can be reflected through using OpenCV with deep learning models like TFLite and mediapipe to understand what activity 
a person is indluged in and to control the luminance of a bulb used to light the given environment. This can not only prove really helpful 
to the person using the bulb and save him efforts in adjusting the light himself but it can also prove really helpful by cutting the excess
energy that would rather be spent on unnecessary extra lighting. 

This model can predict activities like using a phone, laptop or even reading a book and adjust the lighting accordingly. For example using a
phone requires low light, while reading required brighter light, so it adjusts the lighting accordingly.

The whole project was done on a Raspberry Pi 4 as we required a high performance Iot device to control the lights. We integrated the use of
the above specified deep learning models for object detection and pose detection to understand what the person is doing. Then we used the 
library called gpiozero to use the pins of the raspberry pi 4 to control LEDs to show for the proof of concept.

The following gif can show you our proof of concept.
