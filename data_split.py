import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import shutil
path_origin = 'data/TibetanMnist'
path_origin1 = 'data/TibetanMnist'
files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(path_origin)))

random.shuffle(files)
rate = int(len(files) * 0.8)
train_txt = open('data/train.txt','w')
test_txt = open('data/test.txt','w')
for i,f in enumerate(files):
    print(f,f.split('_')[0])
    target_image = os.path.join(path_origin1,f)
    target_label = f[0]
    print(target_image,target_label)
    if i < rate: 
        train_txt.write(target_image + ' ' + target_label+ '\n')
    else:
        test_txt.write(target_image + ' ' + target_label+ '\n')