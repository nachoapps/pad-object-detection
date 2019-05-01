# Training on Google Cloud Vision

The script in this directory can be used to create a dataset for training an
edge model using Google Cloud Vision AutoML.

Unfortunately I'm writing these docs quite a while after actually doing this =(

Fortunately I think I did an OK job documenting this at the time in the script.

First, you should check out the edge device model quickstart:
https://cloud.google.com/vision/automl/docs/edge-quickstart

Follow the instructions; when you get to the section about copying sample data,
come back here and prepare to run the script.

Check out the docs at the top of the script for how to launch it; after the
script runs, it will print out instructions on how to copy the data into the
correct location based on your arguments.

There is more documentation on training the model here:
https://cloud.google.com/vision/automl/docs/train-edge

I've pretrained a model that I'm using in Miru for ^dawnglare. You can find it
here: https://drive.google.com/drive/folders/1RIZaDYEB6HbYv9iDP4EYLsozQl6blSVF