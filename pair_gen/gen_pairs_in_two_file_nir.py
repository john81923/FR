import os, sys
from glob import glob, glob1
import itertools
import numpy as np
from tqdm import tqdm
import random
from os.path import basename, dirname
import pickle
import joblib
import cv2


def load_folder( pair_folder_name, same_txtfile_name, diff_txtfile_name, image_size=(112,112)):
    #bins, issame_list = pickle.load(open(bins_filename, 'rb'))
    #bins, issame_list = pickle.load(open(bins_filename, 'rb', encoding='latin1'))
    flipflag = [0] # if flip use [0,1]
    image_list = []
    with open( os.path.join(pair_folder_name, same_txtfile_name+'.txt' ),'r') as fp:
        for line in fp.readlines():
            line = line.split('\t')
            image_list.append([line[0], line[1], 1])
            #print(image_list)

    with open(os.path.join(pair_folder_name, diff_txtfile_name+'.txt' ),'r') as fp:
        for line in fp.readlines():
            line = line.split('\t')
            image_list.append([line[0], line[1], 0])

    #print("image_list: ", len(image_list))
    #print("image_list[6000]: ", image_list[6000])

    np.random.shuffle(image_list)

    data_list = []
    issame_list = []
    for flip in flipflag:
        data = np.empty((len(image_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(image_list)):
        img1_path = image_list[i][0]
        img2_path = image_list[i][1]
        #print(img1_path)
        #print(img2_path)
        issame = image_list[i][2]
        #print(issame)
        issame_list.append(issame)
        #sys.exit(1)
        #print ("img1_path: ", img1_path)
        #print ("img2_path: ", img2_path)
        #print ("issame: ", issame)
        #img0 = cv2.imdecode()
        img1_path = img1_path.rstrip()
        img2_path = img2_path.rstrip()
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        #print("img1_path: ", img1_path)
        #print("img2_path: ", img2_path)
        #print("img1_path: ", os.path.isfile(img1_path))
        #print("img2_path: ", os.path.isfile(img2_path.rstrip()))

        if img1.shape[1]!=image_size[0]:
            img1 = cv2.resize(img1, image_size)
        if img2.shape[1]!=image_size[0]:
            img2 = cv2.resize(img2, image_size)

        for flip in flipflag:
            if flip==1:
                img1 = np.flip(img1, axis=1)
                img2 = np.flip(img2, axis=1)
            data_list[flip][2*i+0] = img1
            data_list[flip][2*i+1] = img2
        if i%1000==0:
            print('loading folder', i)

    #print(folder_name.split('/')[-1], data_list[0].shape)
    #"""db
    print( 'total len of data_list:', len(data_list[0]))
    writedb = True
    if writedb:
        print( 'dumping to pickle')
        base_data_list = 'henan_gz_val_data'
        base_issame_list = 'henan_gz_val_issame'

        file_data_list = f'/mnt/sdd/johnnysun/data/FR/henan_gz_val/{base_data_list}_{len(data_list[0])//2000}.pkl'
        file_issame_list = f'/mnt/sdd/johnnysun/data/FR/henan_gz_val/{base_issame_list}_{len(data_list[0])//2000}.pkl'
        with open(file_data_list,'wb') as f:
            #data_list = pickle.load(f)
            pickle.dump(data_list, f,protocol = 4)

        with open(file_issame_list,'wb') as f:
            #issame_list = pickle.load(f)
            pickle.dump(issame_list, f,protocol = 4)
        print("write complete")
        print( 'saved:', file_data_list )
        print( 'saved:', file_issame_list)
    #"""
    return (data_list, issame_list)

#
# gen_pickle
pair_folder_name = '/mnt/sdd/johnnysun/data/FR/henan_gz_val'
#


#
# gen txt pair
bench_mark_dir = '/mnt/sdd/johnnysun/data/FR/Guizhou_NIR_dataset_kfr_0304_trainval/val'
bench_mark2_dir = '/mnt/sdd/johnnysun/data/FR/henan_realign/val'

total_pairs_num = 50000 # total positive/negative pairs
total_sub_pairs_num = [20000, 10000]
same_pair_set_name = f'validate_same_pairs_hn_gz_{total_pairs_num//1000}'
diff_pair_set_name = f'validate_diff_pairs_hn_gz_{total_pairs_num//1000}'
#img_type = '*.jpg'
img_type = '*.png'


#data_list, issame_list = load_folder(pair_folder_name, same_pair_set_name, diff_pair_set_name )
#ssys.exit(1)

#total_pairs_num = 2100 # total positive/negative pairs
#positive_pairs_per_person num is:  14003
#True pair: 14366437
# Total folders in
people_gui = sorted( glob(f'{bench_mark_dir}/*') )
prople_hen = sorted( glob(f'{bench_mark2_dir}/*') )
people = people_gui + prople_hen

num_of_people = len(people)
#print("people[0]: ", people[0])
print("num_of_people: ", num_of_people) # 1026 people

same_pairs = []
diff_pairs = []
# positive_pairs_per_person need integer.
positive_pairs_per_person = total_pairs_num // 2 // num_of_people + 1 # divided by 2 is for positive and negative samples
print("positive_pairs_per_person: ", positive_pairs_per_person) # 49

# same pair
for person in tqdm(people):
    imgs = sorted( glob( f'{person}/*.png'))
    #print("imgs[0]: ", imgs[0]) # imgs[0]:  indoor_1001_20_M_Y_C1_E0_G_0p30m_0000.png
    #print("len(imgs): ", len(imgs)) #len(imgs):  168 (images for a person)
    #new_imgs = sorted(glob.glob1(os.path.join(bench_mark_dir, person), img_type))
    #for i in range(len(imgs)):

    #    imgs[i] = int(imgs[i].split('_')[-1][:-4]) #generate the id of image
    #print("new_imgs: ", new_imgs)
    #print("imgs: ", imgs)
    comb = np.asarray(list(itertools.combinations(imgs, 2)))
    #print("comb[0]: ", comb[0]) #comb[0]:  ['indoor_1001_20_M_Y_C1_E0_G_0p30m_0000.png', indoor_1001_20_M_Y_C1_E0_G_0p30m_0001.png']

    #print("length for combination: ", len(list(itertools.combinations(imgs, 2))))
    #exit(0)
    # Shuffle the combinations.
    np.random.shuffle(comb)
    #same_pair.extend(['%s\t%d\t%d' % (person, item[0], item[1]) for item in comb[:positive_pairs_per_person]])
    #print("person: ", person)
    #exit(0)
    #same_pairs.extend([ [person+'/'+item[0], person+'/'+item[1]] for item in comb[:positive_pairs_per_person]])
    same_pair_nir = []
    same_pair_rgb = []
    if 'NIR' not in basename(person):
        for item in comb:
            if "NIR" in basename(item[0]) and "NIR" in basename(item[1]):
                same_pair_nir.append( f'{item[0]}\t{item[1]}' )
            elif "RGB" in basename(item[0]) and "RGB" in basename(item[1]):
                same_pair_rgb.append( f'{item[0]}\t{item[1]}' )
            else:
                continue
        #print(same_pair_nir)
        #print(same_pair_rgb)
        same_pairs.extend( same_pair_nir[:positive_pairs_per_person//2])
        same_pairs.extend( same_pair_rgb[:(positive_pairs_per_person-positive_pairs_per_person//2)])
    else:
        same_pairs.extend( [ f'{item[0]}\t{item[1]}' for item in comb[:positive_pairs_per_person] ] )


np.random.shuffle(same_pairs)
same_pair_num = len(same_pairs)
#positive_pairs_per_person = int(len(same_pairs) / num_of_people + 1)
print("positive_pairs_per_person num is: ", positive_pairs_per_person)
print('True pair:', len(same_pairs)) # 50274


#print("same_pair[0]: ", same_pair[0])
#['1001_NIR/indoor_1001_20_M_Y_C1_E1_N_0p30m_0003.png', '1001_NIR/outdoor_1001_20_M_Y_C2_E2_N_0p50m_0000.png']
#exit(0)

# diff pair

# get all the combinations
comb = list(itertools.combinations(people, 2))
print('diff pair comb. len', len(comb))
negative_img_pair_per_id_pair = total_pairs_num//2//len(comb) +1
print( 'negative_img_pair_per_id_pair', negative_img_pair_per_id_pair)
tmp_list = np.asarray(comb)
diff_rgb_pair = []
diff_nir_pair = []
nir_count = 0
rgb_count = 0
for item in tqdm(tmp_list):
    #for i in range(negative_img_pair_per_id_pair):

    img1_list = glob( f'{item[0]}/*.png' )
    img1 = img1_list[np.random.randint(len(img1_list))]
    img2_list = glob( f'{item[1]}/*.png')
    img2 = img2_list[np.random.randint(len(img2_list))]
    #print(item[0], item[1])
    if 'NIR' in basename( item[0] ):
         nir_rgb = True
    else:
        nir_rgb = random.random()<0.5
    #print('nir_rgb', nir_rgb )

    if nir_rgb:
        #print('img1',basename(img1))
        #print('item',  basename(item[0] ))
        while 'NIR' not in basename(img1) and 'NIR' not in basename(item[0]):
            #print( basename(img1) )
            img1 = img1_list[np.random.randint(len(img1_list))]
        while 'NIR' not in basename(img2) and 'NIR' not in basename(item[1]):
            img2 = img2_list[np.random.randint(len(img2_list))]
        nir_count += 1
        diff_nir_pair.append( f'{img1}\t{img2}'  )
        #print( img1)
    else:

        while 'RGB' not in basename(img1):
            img1 = img1_list[np.random.randint(len(img1_list))]
        while 'RGB' not in basename(img2):
            img2 = img2_list[np.random.randint(len(img2_list))]
        rgb_count += 1
        diff_rgb_pair.append( f'{img1}\t{img2}'  )
        #print( img2)
    #sys.exit(1)
    #img1 = int(img1.split('_')[-1][:-4])
    #img2 = int(img2.split('_')[-1][:-4])
    #diff_pairs.append( [item[0]+'/'+img1, item[1]+'/'+img2] )
    #diff_pairs.append( f'{img1}\t{img2}' )
    #print( diff_pairs)
print('rgb_count', rgb_count)
print('nir_count', nir_count )

np.random.shuffle(diff_rgb_pair)
np.random.shuffle(diff_nir_pair)

# After shuffle, select the same_pair_num from the beginning.
diff_pairs = diff_rgb_pair[:same_pair_num//2] + diff_nir_pair[:(same_pair_num-same_pair_num//2)]


print('False pair:', len(diff_pairs))
print("Total pairs num is: ", len(same_pairs)+len(diff_pairs))
#sys.exit(1)

'''
start = 0
n = 1
while start < len(comb):
    print('start', )
    tmp_list = np.asarray( comb[start:start+num_of_people-n-1] )
    #print("tmp_list: ", tmp_list)
    #print("len(tmp_list): ", len(tmp_list))
    #if n == 2:
    #    exit(0)
    np.random.shuffle(tmp_list)
    #print("n: ", n)
    #print("positive_pairs_per_person: ", positive_pairs_per_person)
    #print("len(tmp_list): ", len(tmp_list))

    #for item in tmp_list[:positive_pairs_per_person+1]: #49+1 = 50
    for item in tmp_list[:positive_pairs_per_person+40]: #51+40
        #try:

            #print ("(positive_pairs_per_person/len(tmp_list)): ", (positive_pairs_per_person/len(tmp_list)))
            #exit (0)

        img1 = glob( f'{item[0]}/*.png' )
        img1 = img1[np.random.randint(len(img1))]
        #img1 = int(img1.split('_')[-1][:-4])
        img2 = glob( f'{item[1]}/*.png')
        #print img2, item[1]

        img2 = img2[np.random.randint(len(img2))]
        #img2 = int(img2.split('_')[-1][:-4])
        #diff_pairs.append( [item[0]+'/'+img1, item[1]+'/'+img2] )
        diff_pairs.append( f'{img1}\t{img2}' )
        #print( f'{item[0]}\t{item[1]}'  )

        #diff_pair.append('%s\t%d\t%s\t%d' % (item[0], img1, item[1], img2))
        #diff_pair:  ['1001_NIR/indoor_1001_20_M_Y_C1_E0_N_0p50m_0012.png', '1002_NIR/indoor_1296_20_F_Y_C1_E0_N_0p50m_0003.png']
        #print ("diff_pair: ", diff_pair[0])
        #exit(0)
        #except:
        #    continue
    start += num_of_people - n
    n += 1

# Shuffle diff pairs
np.random.shuffle(diff_pairs)
# After shuffle, select the same_pair_num from the beginning.
diff_pairs = diff_pairs[:same_pair_num]

print('False pair:', len(diff_pairs))
print("Total pairs num is: ", len(same_pairs)+len(diff_pairs))
sys.exit(1)
'''
#print('True pair:', len(same_pair), ', False pair:', len(diff_pair))
#exit(0)
#pairs = same_pair + diff_pair
#np.random.shuffle(pairs)
'''
with open(data_set_name+'.txt', 'w') as f:
    f.write('10\t300\n')
with open(data_set_name+'.txt', 'w') as f:
    f.write('\n'.join(pairs))
'''


with open( f'{pair_folder_name}/{same_pair_set_name}'+'.txt', 'w') as f:
 f.write('\n'.join(same_pairs))

with open( f'{pair_folder_name}/{diff_pair_set_name}'+'.txt', 'w') as f:
 f.write('\n'.join(diff_pairs))

data_list, issame_list = load_folder(pair_folder_name, same_pair_set_name, diff_pair_set_name )



total_sub_pairs_num = [20000, 10000]
np.random.shuffle(same_pairs)
np.random.shuffle(diff_pairs)
for subp in total_sub_pairs_num:
    same_pair_set_name = f'validate_same_pairs_hn_gz_{subp//1000}'
    diff_pair_set_name = f'validate_diff_pairs_hn_gz_{subp//1000}'

    with open( f'{pair_folder_name}/{same_pair_set_name}'+'.txt', 'w') as f:
        f.write('\n'.join(same_pairs[:subp//2]))
    with open( f'{pair_folder_name}/{diff_pair_set_name}'+'.txt', 'w') as f:
        f.write('\n'.join(diff_pairs[:subp//2]))

    data_list, issame_list = load_folder(pair_folder_name, same_pair_set_name, diff_pair_set_name )





#
