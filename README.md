# Emotion Recognition - 100428904
Y3P Project by Christian Iannotta

## Overview
This project detects a human face in an image and attempts to classify the image based on the emotion displayed on the face

## Pipeline
To achieve the end goal, I have broken the problem down into distinct modular stages, creating a pipeline as shown below:
NOTE: whilst this project remains in development, each stage will be marked with a "X" when completed and fully functional

                                    -   recieve frame from zoom []
                                    1)  detect face and create face mesh []
                                    2)  validation and quality checks []
                                        -   fallback to previous frame if needed
                                    3)  find bounding box for face []
                                        -   crop raw frame to new ROI
                                    4)  align and normalise []
                                    5)  feature extraction []
                                    6)  classification []
                                    -   output []

...  IN PROGRESS  ...
