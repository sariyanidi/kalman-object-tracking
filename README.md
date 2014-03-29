Multiple object tracking code based on OpenCV library.

OpenCV 2.2+ is needed to run code. Main.cpp is not part of this code, but its just given to show the usage of Tracker class, its quite simple.

This application is supposed to run on video data. Objects detected at each frame are given to Tracker class, and this class trackes detections which point to the same object.
The tracking rectangle is decided using Kalman filtering.
