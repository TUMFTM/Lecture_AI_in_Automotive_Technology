# Artificial Intelligence in Automotive Technology - Lecture 2

In the practice session of lecture 2 we will define a special computer vision pipeline. The goal of the pipeline is to detect lane lines in images and video. We are using different images to check if we worked correctly and we are applying our Computer Vision Knowhow from lecture 2.

## Define The Pipeline

First of all we have to define the pipeline for processing the image. We are splitting the code in two parts
* A function which is called “process_image”
* A main software part which is loading the image
In addition, for a better overview we are moving all functions for processing the image to a library called “functions.py”
