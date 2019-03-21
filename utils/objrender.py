'''
This is a script for rendering png pictures for 3d models in ShapeNet using blender
It will render NUM png pictures from arbitrary angle for each 3d model 
with transparent background

Author: Ziwen Chen
'''

import bpy
import os
import random
import math
import numpy as np

NUM = 5

models_folder = '/home/chenziwe/sceneslicer/SceneSlicer/dataset/ShapeNetSubset/'
synsets = next(os.walk(models_folder))[1]

save_folder = '/home/chenziwe/sceneslicer/SceneSlicer/dataset/ShapeNetRendered/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

#place the camera facing origin at distance from arbitrary location
def update_camera(camera, distance=1.2):
    pi = 3.14159265
    a = random.randint(0,360)/180*pi
    b = random.randint(0,360)/180*pi
    y = random.randint(0,360)/180*pi
    
    sina = math.sin(a)
    cosa = math.cos(a)
    sinb = math.sin(b)
    cosb = math.cos(b)
    siny = math.sin(y)
    cosy = math.cos(y)
    
    xmat = np.array([[1.,0,0], [0,cosa,-sina],[0,sina,cosa]])
    ymat = np.array([[cosb,0,sinb], [0,1,0],[-sinb,0,cosb]])
    zmat = np.array([[cosy,-siny,0], [siny,cosy,0],[0,0,1]])
    
    # unit vector indicating the direction the camera pointing to
    direction = np.array([0.,0,-1])
    direction = np.matmul(xmat,direction)
    direction = np.matmul(ymat,direction)
    direction = np.matmul(zmat,direction)
    
    location= -direction*distance
    
    # Set camera rotation in euler angles
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler[0] = a
    camera.rotation_euler[1] = b
    camera.rotation_euler[2] = y
    # Set camera translation
    camera.location.x = location[0]
    camera.location.y = location[1]
    camera.location.z = location[2]


context = bpy.context
data = bpy.data

#scene and rendering setting
context.scene.render.engine = 'BLENDER_EEVEE'
context.scene.render.alpha_mode = 'TRANSPARENT'
context.scene.render.image_settings.file_format = 'PNG'
context.scene.render.image_settings.color_mode = 'RGBA'
context.scene.display_settings.display_device = 'sRGB'
context.scene.view_settings.view_transform = 'RRT'
context.scene.render.image_settings.color_depth = '16'
#context.scene.view_settings.exposure = 2.5
context.scene.view_settings.gamma = 2.2
context.scene.render.resolution_x = 128
context.scene.render.resolution_y = 128

camera = data.objects['Camera']

# loop through all synsets
for s in synsets:
    save_synset_path = save_folder+s+'/'
    if not os.path.exists(save_synset_path):
        os.mkdir(save_synset_path)
    synset_path = models_folder+s+'/'
    # loop through the models in a synset
    models = next(os.walk(synset_path))[1]
    for m in models:
        save_model_path = save_synset_path+m+'/'
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        model_path = synset_path+m+'/models/model_normalized.obj'
        bpy.ops.import_scene.obj(filepath=model_path)
        object = data.objects[0]
        object.location.x=0
        object.location.y=0
        object.location.z=0
        for i in range(NUM):
            e = random.randint(0,10)/10
            context.scene.view_settings.exposure = 2+e
            update_camera(camera)
            context.scene.render.filepath = save_model_path+str(i)+'.png'
            bpy.ops.render.render(write_still=True)
        bpy.ops.object.delete()


