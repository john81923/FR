import os
import glob
import itertools
import numpy as np

#bench_mark_dir = '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr/'

#bench_mark_dir = '/mnt/atdata/FacialRecognition/FR_Age/AsianTestDatasets_badpose_val'

bench_mark_dir = '/mnt/sdc/craig/Asia_face_dataset/FR_TPE_Street_Video_Labeled_Dataset_badpose_cleaned'

same_pair_set_name = 'validate_same_pairs_tpe_street'
diff_pair_set_name = 'validate_diff_pairs_tpe_street'
#same_pair_set_name = 'validate_same_pairs_asia_age'
#diff_pair_set_name = 'validate_diff_pairs_asia_age'
img_type1 = '*.jpg'
img_type2 = '*.png'
total_pairs_num = 30000 # total positive/negative pairs
#total_pairs_num = 2100 # total positive/negative pairs
#positive_pairs_per_person num is:  14003
#True pair: 14366437
# Total folders in
people = sorted(os.listdir(bench_mark_dir))
num_of_people = len(people)
#print("people[0]: ", people[0])
print("num_of_people: ", num_of_people) # 1026 people
same_pairs = []
diff_pairs = []
# positive_pairs_per_person need integer.
positive_pairs_per_person = total_pairs_num // 2 // num_of_people + 1 # divided by 2 is for positive and negative samples

#print("positive_pairs_per_person: ", positive_pairs_per_person) # 49
# same pair
for person in people:
    imgs = glob.glob1(os.path.join(bench_mark_dir, person), img_type1)
    imgs.extend(glob.glob1(os.path.join(bench_mark_dir, person), img_type2))
    imgs = sorted(imgs)

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
    same_pairs.extend(['%s/%s\t%s/%s' % (person, item[0], person, item[1]) for item in comb[:positive_pairs_per_person]])
    #same_pairs.extend([ [person+'/'+item[0], person+'/'+item[1]] for item in comb])
    #break

np.random.shuffle(same_pairs)
same_pair_num = len(same_pairs)
#positive_pairs_per_person = int(len(same_pairs) / num_of_people + 1)
print("positive_pairs_per_person num is: ", positive_pairs_per_person)
print('True pair:', len(same_pairs)) # 50274


#print("same_pair[0]: ", same_pair[0])
#['1001_NIR/indoor_1001_20_M_Y_C1_E1_N_0p30m_0003.png', '1001_NIR/outdoor_1001_20_M_Y_C2_E2_N_0p50m_0000.png']
#exit(0)

# diff pair
start = 0
n = 1
# get all the combinations
comb = list(itertools.combinations(people, 2))

while start < len(comb):
    tmp_list = np.asarray(comb[start:start+num_of_people-n-1])
    #print("tmp_list: ", tmp_list)
    #print("len(tmp_list): ", len(tmp_list))
    #if n == 2:
    #    exit(0)
    np.random.shuffle(tmp_list)
    #print("n: ", n)
    #print("positive_pairs_per_person: ", positive_pairs_per_person)
    #print("len(tmp_list): ", len(tmp_list))

    #for item in tmp_list[:positive_pairs_per_person+1]: #49+1 = 50
    #for item in tmp_list[:positive_pairs_per_person+40]: #51+40
    for item in tmp_list[:positive_pairs_per_person+1]: #51+40
        #try:

            #print ("(positive_pairs_per_person/len(tmp_list)): ", (positive_pairs_per_person/len(tmp_list)))
            #exit (0)
        img1 = glob.glob1(os.path.join(bench_mark_dir, item[0]), img_type1)
        img1.extend(glob.glob1(os.path.join(bench_mark_dir, item[0]), img_type2))

        img2 = glob.glob1(os.path.join(bench_mark_dir, item[1]), img_type1)
        img2.extend(glob.glob1(os.path.join(bench_mark_dir, item[1]), img_type2))
        if len (img1) >0 and len (img2) > 0:
            img1 = img1[np.random.randint(len(img1))]
            #img1 = int(img1.split('_')[-1][:-4])
            #print img2, item[1]
            img2 = img2[np.random.randint(len(img2))]
            #img2 = int(img2.split('_')[-1][:-4])
            #diff_pairs.append( [item[0]+'/'+img1, item[1]+'/'+img2] )
            diff_pairs.append('%s/%s\t%s/%s' % (item[0], img1, item[1], img2))

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


with open(same_pair_set_name+'.txt', 'w') as f:
 f.write('\n'.join(same_pairs))

with open(diff_pair_set_name+'.txt', 'w') as f:
 f.write('\n'.join(diff_pairs))
