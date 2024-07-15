import cv2
import os
from PIL import Image
import numpy as np

result_path = "output_masks_nopost_M_permask"

out_folder = "output_masks_nopost_M_permask_video"
print("out_folder   ",out_folder)
os.makedirs(out_folder,exist_ok=True)

img_names = sorted(os.listdir(result_path))
fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
# size = (3840,720)
size = (593,816)
fps = float(3)
writer = cv2.VideoWriter(os.path.join(out_folder,'result.mp4'), fourcc, fps, size)


for i in range(len(img_names)):
    img_name = 'point_'+str(i)+'.png'
    img_path = os.path.join(result_path,img_name)
    print(img_path)
    img = np.array(Image.open(img_path))
    img = img[:,:,:3]

    writer.write(img[:,:,::-1])
writer.release()
import pdb;pdb.set_trace()
        



