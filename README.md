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

## Demo
<img src="demo/1.jpg" alt="1" width="200" height="200" />
<img src="demo/11.jpg" alt="11" width="200" height="200" />

<img src="demo/2.jpg" alt="2" width="200" height="200" />
<img src="demo/22.jpg" alt="22" width="200" height="200" />

![3](demo/3.jpg)
![33](demo/33.jpg)
![333](demo/333.jpg)

![4](demo/4.jpg)
![44](demo/44.jpg)
