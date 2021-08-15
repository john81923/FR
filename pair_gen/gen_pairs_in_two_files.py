import os
import glob
import itertools
import numpy as np
from tqdm import tqdm

bench_mark_dir = './validate_faces'
same_pair_set_name = 'validate_same_pairs_test'
diff_pair_set_name = 'validate_diff_pairs_test'
img_type = '*.bmp'

# Total folders in
people = sorted(os.listdir(bench_mark_dir))
num_of_people = len(people)

same_pairs = []
diff_pairs = []

# same pair
for person in people:
    imgs = sorted(glob.glob1(os.path.join(bench_mark_dir, person), img_type))
    if len(imgs) == 0:
        continue
    comb_img = np.asarray(list(itertools.combinations(imgs, 2)))

    # Shuffle the image combinations in the same ID.
    np.random.shuffle(comb_img)
    same_pairs.extend(['%s/%s\t%s/%s' % (person, item[0], person, item[1]) for item in comb_img])

# Shuffle same pairs
np.random.shuffle(same_pairs)
same_pair_num = len(same_pairs)

positive_pairs_per_person = int(len(same_pairs) / num_of_people + 1)
print("positive_pairs_per_person num is: ", positive_pairs_per_person)
print('True pair:', len(same_pairs))

# diff pair
start = 0
n = 1
# combination of people
comb = list(itertools.combinations(people, 2))

while start < len(comb):
    tmp_list = np.asarray(comb[start:start+num_of_people-n-1])
    np.random.shuffle(tmp_list)
    for item in tmp_list[:positive_pairs_per_person+1]:
        try:
            img1 = sorted(glob.glob1(os.path.join(bench_mark_dir, item[0]), img_type))
            if len(img1) == 0:
                continue
            # rand images
            img1 = img1[np.random.randint(len(img1))]

            img2 = sorted(glob.glob1(os.path.join(bench_mark_dir, item[1]), img_type))
            if len(img2) == 0:
                continue
            # rand images
            img2 = img2[np.random.randint(len(img2))]

            diff_pairs.append('%s/%s\t%s/%s' % (item[0], img1, item[1], img2))
        except:
            continue
    start += num_of_people - n
    n += 1

"""
for item in tqdm(comb):
    try:
        img1 = sorted(glob.glob1(os.path.join(bench_mark_dir, item[0]), img_type))
        img2 = sorted(glob.glob1(os.path.join(bench_mark_dir, item[1]), img_type))

        img_comb = np.asarray(list(itertools.product(img1, img2)))
        img_comb = itertools.product(img1, img2)

        for item_img_comb in img_comb:
            diff_pairs.append('%s/%s\t%s/%s' % (item[0], item_img_comb[0], item[1], item_img_comb[1]))
    except:
        continue
"""

# Shuffle diff pairs
np.random.shuffle(diff_pairs)
# After shuffle, select the same_pair_num from the beginning.
diff_pairs = diff_pairs[:same_pair_num]

print('False pair:', len(diff_pairs))
print("Total pairs num is: ", len(same_pairs)+len(diff_pairs))

with open(same_pair_set_name+'.txt', 'w') as f:
    f.write('\n'.join(same_pairs))

with open(diff_pair_set_name+'.txt', 'w') as f:
    f.write('\n'.join(diff_pairs))
