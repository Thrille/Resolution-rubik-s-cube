import pybullet as pb
import pybullet_data
import time
import os, glob, random
import cv2
from PIL import Image
import numpy as np
import io
import math

physicsClient = pb.connect(pb.GUI)

pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF('plane.urdf')

cube2 = pb.loadURDF('cube.urdf', [0,0,0])
# texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)
# random_texture_path = texture_paths[random.randint(0,len(texture_paths)-1)]
# textureId = pb.loadTexture(random_texture_path)
# pb.changeVisualShape(cube, -1, textureUniqueId=textureId)

pb.setGravity(0, 0, -9.8)
pb.resetDebugVisualizerCamera(1.40, -53.0, -39.0, (0.53, 0.21, -0.24))

pb.setRealTimeSimulation(1)

viewMatrix = pb.computeViewMatrix(
    cameraEyePosition=[0.25,0,0],
    cameraTargetPosition=[0,0,0],
    cameraUpVector=[0,1,0])

projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=3.1)

width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
    width=224, 
    height=224,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix)

print(rgbImg)

cube2.jointNameToId['cube_cube2']

cv2.imshow('test', rgbImg)
rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
cv2.imwrite('my.png',rgbImg)
# image = np.array(Image.open(io.BytesIO(rgbImg))) 
# data = np.zeros((224, 224, 4), dtype=np.uint8)
# data[223,223] = image
# img = Image.fromarray(data, 'RGB')
# img.save("my.png")
# img.show()

time.sleep(50)