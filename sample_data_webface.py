from glob import glob
import random
import sys
import os

data_path = '/mnt/sdc/johnnysun/data/FR/WebFace260M-Full/WebFace260M_realign'
data_list = glob(f'{data_path}/*')

sample_number = 20
random.shuffle(data_list)

for data_dir in data_list[:sample_number]:
    os.system(f'cp -r {data_dir} /mnt/sdc/johnnysun/data/FR/WebFace260M-Full/sample/')
