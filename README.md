# PAD Machine Learning Data

I created this repo as an experiment in using machine learning to identify
items on screen in Puzzle and Dragons.

## TLDR

It works pretty well. I'm providing a few example models I built, and scripts
for you to regenerate them or train your own. 

Here are some screenshots of the faster_rcnn model after about two hours of
training.

<div align="center">
 <img src="https://raw.githubusercontent.com/nachoapps/pad-object-detection/master/docs/phone_trained.png" height="600px">
 <img src="https://raw.githubusercontent.com/nachoapps/pad-object-detection/master/docs/screen_trained.png" height="600px">
</div>

If you just want to play with the frozen models, I'll put a couple
[here](https://drive.google.com/drive/folders/1RIZaDYEB6HbYv9iDP4EYLsozQl6blSVF).


### Why use ML

I have previously tried some hand-crafted systems, like using corner detection
to identify where the portrait boxes are, and using histogram analysis to
identify orbs.

Those attempts kind of worked; Miru Bot currently uses closest histogram to
identify orbs for ^dawnglare. This has some limitations, namely that it is
difficult to correctly box out the orbs for identification across a wide
range of devices, and even when correct it seems to fail occasionally.

The advantage is that it reasonably fast to execute, easy to understand,
and to augment.

The disadvantage is that there's really nothing to do when it randomly
fails; adding additional examples can make existing matches worse.

Also, you can't just find arbitrary things in the image, you have to figure
out an identification technique for each of them.

This approach uses box annotations and the Tensorflow
[Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
library to try to pick out items in the image.

The main disadvantages are the effort required to box everything you want to
find, difficulty of setting up Tensorflow, and potentially the resulting
model size and execution time. 

## Example images

I've provided some sample boards in images (some mine, some donated by users 
on the PAD Community Discord). There are a couple of different types:

* Screenshots of the game running, on a variety of phones. I intend to use
these for identifying screenshots via ^dawnglare.
* Screen captures of streamers playing PAD (only a couple, cropped). This might
be useful for finding boards in random images, like for a Twitch.tv plugin.
* Images taken via a camera. Mostly upright, some examples of different tilts.
I want to use this eventually for my mechanical PAD solver.

The images are kind of large because I didn't think to scale them down before
annotating, oops. The tfrecord conversion script applies a scaling before
saving the records.

You can download them
[here](https://drive.google.com/open?id=1kk46jpebpeBFkYX18iD8D6eP8WKE_wFz).
Stick them in the [images](images/images.md) directory.

## Labeling Images

I used this tool to create annotations: https://github.com/tzutalin/labelImg

Works fine on both Windows (via the prebuilt binary) and Linux (checked out the
repo and built it).

I've created the following annotations:

* 5x4board
* 6x5board
* 7x6board
* board (just overlaps the X/Y board annotation)
* rorb (red/fire)
* borb (blue/water)
* gorb (green/wood)
* lorb (light)
* dorb (dark)
* jorb (jammer)
* porb (poison)
* morb (mortal poison)
* lock (identifies just the lock)
* plus (identifies just the plus)
* portrait (team monster boxes)
* horb (heart)
* oorb (bomb)

I identified every example of every one of these in every image. It was 
exhausting and I probably got a little sloppy towards the end. Feel free to
correct any bad annotations and send me a pull request.


## Setup for running the scripts

### Configuring Python

I went with Python 3.5 configured via virtualenv, per Tensorflow's 
recommendation.

I installed all the deps mentioned in the various guides in my tensorflow
virtualenv.


### Installing Tensorflow

Follow the instructions here:
https://www.tensorflow.org/install/install_linux

For Tensorflow 1.4 be sure to install CUDA v8 and cuDNN v6 EXACTLY. Trying to
mix and match is like asking to suffer. 

# Set up tensorflow/models/object_detection

Follow the instructions here:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# Get started

Check out the [commands](scripts/README.md) I've provided. These allow you to
generate the training data, train the model, run evaluation, export a frozen
model, and run inference on an arbitrary image.
