
from os.path import dirname, basename
from glob import glob

tar_path = '/mnt/sdd/johnnysun/data/FR/henan_realign/val'

id_list = glob(f'{tar_path}/*')
id_list.sort()
for id in id_list:
    id_img = glob(f'{id}/*')
    if len(id_img)<1000:
        print( basename( id), len(id_img))
