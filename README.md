# SceneSlicer
Scene Slicer is a program to automatically slice a 2D scene image into multiple layers of objects and background.

When given an image, the slicer automatically identifies foreground objects, masks them out, and inpaints the hole with reasonable imagined background.

## Author

Ziwen Chen

## Purpose
The purpose of Scene Slicer is to facilitate 3D scene recontruction -- by cleaning up the clutters in the foreground, 
we can make the geometry of the underlying environment (e.g. an indoor room) more apparent, 
and thus make it easier for a scene resontructor to fathom the shape of the environment. 
After the global coordinate is pinned down, foreground objects can be more easily localized inside the global coordinate.

## Method

Currently, I'm using Mask RCNN combined with DeepFill as the backbone of Scene Slicer. 
So far, RCNN-based networks are still in leading position regarding to accuracy compared to other kinds of object detectors.
DeepFill is a state-of-the-art background inpainter. 
I trained both networks using MSCOCO dataset from scatch.

## Current Progress

The inpainter still needs some improvement it seems...

## Demo
<img src="demo/1.jpg" alt="1" width="200" height="200" /> <img src="demo/11.jpg" alt="11" width="200" height="200" />

<img src="demo/2.jpg" alt="2" width="200" height="200" /> <img src="demo/22.jpg" alt="22" width="200" height="200" />

<img src="demo/3.jpg" alt="3" width="200" height="200" /> <img src="demo/33.jpg" alt="33" width="200" height="200" /> <img src="demo/333.jpg" alt="333" width="200" height="200" />

<img src="demo/4.jpg" alt="4" width="200" height="200" /> <img src="demo/44.jpg" alt="44" width="200" height="200" />
