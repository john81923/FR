import numpy as np
from keras.utils import np_utils
from PIL import Image
import random
import os
import glob

def load_data_flip(image_paths, target_size, box=None):
    """
    Given list of paths, resize and bounding box load images as one numpy array of shape
        (num_images, crop_size, crop_size, channel)
        box:[top, left, bottom, right]
    :return X: image array
     return y: one hot encoded labels
    """
    X = np.zeros((len(image_paths), target_size[0],target_size[1], 3))
    if box:
        ## google output box :## 0: top 1: left 2 lower 3 right
        for i,path in enumerate(image_paths):
            new_box = (box[i][1],box[i][0],box[i][3], box[i][2])
            if bool(random.getrandbits(1)):
                X[i, :] = np.asarray(load_img(path, target_size=target_size, box=new_box))
            else: 
                X[i, :] = np.asarray(load_img(path, target_size=target_size, box=new_box))[:,::-1,:]
        y = np_utils.to_categorical(labels, num_of_class)
        return X, y
    else:
        for i,path in enumerate(image_paths):
            if bool(random.getrandbits(1)):
                X[i, :] = np.asarray(load_img(path, target_size=target_size))
            else: 
                X[i, :] = np.asarray(load_img(path, target_size=target_size))[:,::-1,:]
        return X


def load_img(path, grayscale=False, target_size=None, box=None):
    if Image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = Image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if box:
        img = crop_img(img, box)
        
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        #resize(w, h)
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


def normalization(X):
    return X / 255. - 0.5
def inverse_normalization(X):
    return (X + 0.5)
    
    
def generate_arrays(paths, labels, batch_size, rand=True, target_size = (224,224)):
    sample_number = len(paths)
    while True:
        index = list(range(sample_number))
        if rand:
            np.random.shuffle(index)
        for i in range(0,sample_number,batch_size):
            idx = index[i:i+batch_size]
            X_train = load_data_flip(paths[idx], target_size=target_size)
            y_train = labels[idx]
            X_train = normalization(X_train)
            yield (X_train, y_train)
            
            
def generate_tri(paths, labels, tri_index, batch_size, num_of_class_in_train_dir, target_size = (224,224)):
    sample_number = len(paths)
    anchor_idx, pos_idx, neg_idx = tri_index
    X_train = load_data_flip(paths, target_size=target_size)
    X_train = normalization(X_train)
    while True:
        index = np.arange(sample_number)
        for i in range(0,sample_number,batch_size):
            idx = index[i:i+batch_size]
            anchor_list, pos_list, neg_list = X_train[anchor_idx[idx]], X_train[pos_idx[idx]], X_train[neg_idx[idx]]
            anchor_y, pos_y, neg_y = labels[anchor_idx[idx]], labels[pos_idx[idx]], labels[neg_idx[idx]]
            
            anchor_y = np_utils.to_categorical(anchor_y, num_of_class_in_train_dir)
            pos_y = np_utils.to_categorical(pos_y, num_of_class_in_train_dir)
            neg_y = np_utils.to_categorical(neg_y, num_of_class_in_train_dir)

            yield(
                {
                    'anchor': anchor_list,
                    'pos'   : pos_list,
                    'neg'   : neg_list
                },
                {
                    'triloss': np.zeros(len(anchor_list)),
                    'pred_anchor': anchor_y,
                    'pred_pos': pos_y,
                    'pred_neg': neg_y,
                }
            )
            
            
def fc_layer(bottom, out_size, name, ran=[-1., 1.]):
    in_size = bottom.get_shape()[-1]
    with tf.variable_scope(name):
        weights, biases = get_fc_var(in_size, out_size, name)

        biases = tf.fake_quant_with_min_max_args(biases, min=ran[0], max=ran[1])
        weights = tf.fake_quant_with_min_max_args(weights, min=ran[0], max=ran[1])

        x = tf.contrib.layers.flatten(bottom)
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        fc = tf.fake_quant_with_min_max_args(fc, min=ran[0], max=ran[1])
        return fc


def conv_layer(bottom, out_channels, kernel_size, stride, name, ran=[-1., 1.], activation='relu'):
    in_channels = bottom.get_shape()[-1]
    assert (bottom.get_shape()[-1] == in_channels)
    with tf.variable_scope(name):
        filt, conv_biases = get_conv_var(kernel_size, in_channels, out_channels, name)
        
        filt = tf.fake_quant_with_min_max_args(filt, min=ran[0], max=ran[1])
        conv_biases = tf.fake_quant_with_min_max_args(conv_biases, min=ran[0], max=ran[1])

        conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
        relu = tf.nn.bias_add(conv, conv_biases)
        relu = tf.fake_quant_with_min_max_args(relu, min=ran[0], max=ran[1])
        if activation:
            relu = tf.nn.relu(relu)
        return relu


def get_fc_var(in_size, out_size, name):
    weights = tf.get_variable(name + "_weights",
                              shape=[in_size, out_size],
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
    biases = tf.get_variable(name + "_biases",
                             shape=[out_size],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
    return weights, biases


def get_conv_var(filter_size, in_channels, out_channels, name):
    filters = tf.get_variable(name + "_filters",
                              shape=[filter_size, filter_size, in_channels, out_channels],
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
    biases = tf.get_variable(name + "_biases",
                             shape=[out_channels],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
    return filters, biases

# for facenet

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
    def __len__(self):
        return len(self.image_paths)
    
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = os.listdir(path_exp)
    classes = [item for item in classes if 'txt' not in item]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = glob.glob(os.path.join(facedir,'*'))
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def sample_people(dataset, people_per_batch=40, images_per_person=30):

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    people_per_batch = min(people_per_batch, nrof_classes)
    nrof_images = people_per_batch * images_per_person

    i = 0
    image_paths = []
    image_labels = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1

    return np.asarray(image_paths), num_per_class, np.asarray(sampled_class_indices)

def inf(model, image_paths, img_size, embedding_size, batch_size=30):
    nrof_images = len(image_paths)
    labels_array = np.arange(nrof_images)
    val_gene = generate_arrays(paths=image_paths, rand=False, batch_size=batch_size, labels=labels_array, target_size=img_size[:2])
    emb_array = np.zeros((nrof_images, embedding_size))
    label_check_array = np.zeros((nrof_images,))
    nrof_batches = int(np.ceil(1.0*nrof_images / batch_size))
    for i in xrange(nrof_batches):
        x, y = val_gene.next()
        emb = model.predict(x)
        emb = emb / (np.linalg.norm(emb, axis=-1)+1e-10).reshape(-1,1)
        emb_array[y,:] = emb
        label_check_array[y]=1
    assert(np.all(label_check_array==1))
    return emb_array

def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    
    anchor_list = []
    pos_list = []
    neg_list = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    anchor_list.append(a_idx)
                    pos_list.append(p_idx)
                    neg_list.append(n_idx)
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    #trip_idx += 1
                
                num_trips += 1

        emb_start_idx += nrof_images
    
    idx = np.arange(len(anchor_list))
    np.random.shuffle(idx)
    anchor_list, pos_list, neg_list = np.asarray(anchor_list), np.asarray(pos_list), np.asarray(neg_list)
    anchor_list = anchor_list[idx]
    pos_list = pos_list[idx]
    neg_list = neg_list[idx]
                    
    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets), anchor_list, pos_list, neg_list
