# Emotion Recognition - 100428904
Y3P Project by Christian Iannotta

## Overview
This project detects a human face in an image and attempts to classify the image based on the emotion displayed on the face

## Pipeline
To achieve the end goal, I have broken the problem down into distinct modular stages, creating a pipeline as shown below:
NOTE: whilst this project remains in development, each stage will be marked with a "X" when completed and fully functional

                                    -   recieve frame from zoom []
                                    1)  apply preprocessing [X]
                                    2)  find face and create bounding box [X]
                                    3)  crop raw frame to bounding box []
                                        -   reapply preprocessing to cropped version of raw frame
                                    4)  find facial landmarks []
                                    5)  align and normalise []
                                    6)  classify []
                                    -   output []

...  IN PROGRESS  ...
