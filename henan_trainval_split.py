from glob import glob
import sys
import os
from os.path import basename


data_list = [
        '/mnt/sdd/johnnysun/data/FR/white850_realign',
        '/mnt/sdd/johnnysun/data/FR/black940_realign'
]


# sample 80 each
valnumb = 80

for data_path in data_list:
    valcount = 0
    id_path_list = glob( f'{data_path}/*' )
    for id_path in id_path_list:
        id_name = basename(id_path)
        if id_name == 'train' or id_name == 'val':
            continue
        if valcount< valnumb :
            #print(f'mv {id_path} {data_path}/val/')
            os.system( f'mv {id_path} {data_path}/val/')
        else:
            #print(f'mv {id_path} {data_path}/train/')
            os.system( f'mv {id_path} {data_path}/train/')
        valcount += 1
