""" Code for loading data. """
import os
import cv2
import random
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, metatrain_folders, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        #self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.meta_batch_size = FLAGS.meta_batch_size
        self.meta_iters = FLAGS.metatrain_iterations
        #self.target_iters = 50
        self.target_iters = FLAGS.target_maml_iterations
    
        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.img_size = config.get('img_size', (FLAGS.img_size, FLAGS.img_size))
        self.dim_input = np.prod(self.img_size)*3
        self.dim_output = self.num_classes

        # folder with a bunch of classes -- contain ONLY training samples
        '''metatrain_folder = './data/metatrain/'
            
        # specify how to select classes for experiments
        exp = 'mnist'
        if exp == 'mnist':
            metatrain_folders = [os.path.join(metatrain_folder, label) for label in os.listdir(metatrain_folder) \
                                 if (os.path.isdir(os.path.join(metatrain_folder, label)) and (label in map(str,range(10))))]
        elif exp == 'miniimagenet':
            # metatrain_folders = []
            pass
            
        metatrain_folders = ['./data/metatrain/0','./data/metatrain/1']
        print(metatrain_folders)
        '''

        self.metatrain_character_folders = metatrain_folders
        self.metaval_character_folders = self.metatrain_character_folders #metaval_folders
        self.rotations = config.get('rotations', [0])

    def make_data_tensor(self, train=True, data_folder=None, num_total_batches=-1, target=False):
        if True: #train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            if num_total_batches == -1:
                num_total_batches = 10+self.meta_iters*self.meta_batch_size #200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for bt in range(num_total_batches):
            
            sampled_character_folders = [f for f in folders]
            if bt < (self.meta_iters-self.target_iters)*self.meta_batch_size:
                random.shuffle(sampled_character_folders)
            
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        
        image = tf.image.decode_jpeg(image_file, channels=3)
        image.set_shape((self.img_size[0],self.img_size[1],3))
        image = tf.reshape(image, [self.dim_input])
        image = tf.cast(image, tf.float32) / 255.0
        
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size  * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape:',images.get_shape().as_list())
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]
            
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                #class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def load_target_task(self,class_folders):
        train_dir = './data/metatrain'
        val_dir = './data/metaval'
        channels = 3
        unit = np.diag(np.ones(len(class_folders)))
        def read_img(path):
            im = cv2.cvtColor(cv2.resize(cv2.imread(path),self.img_size),cv2.COLOR_BGR2RGB)
            return np.float32(im).flatten() / 255.0
        ds,vds = [],[]
        for i,cls in enumerate(class_folders):
            cls = os.path.split(cls)[-1]
            cls_train = os.path.join(train_dir,cls)
            samples = list(map(read_img,[os.path.join(cls_train,s) for s in os.listdir(cls_train)]))
            ds.extend(list(zip(samples,[unit[i]]*len(samples))))
            cls_val = os.path.join(val_dir,cls)
            samples = list(map(read_img,[os.path.join(cls_val,s) for s in os.listdir(cls_val)]))
            vds.extend(list(zip(samples,[unit[i]]*len(samples))))
        np.random.shuffle(ds)
        np.random.shuffle(vds)
        x,y = zip(*ds)
        vx,vy = zip(*vds)
        x,vx = np.float32(x),np.float32(vx)
        y,vy = np.int32(y),np.int32(vy)
        return (x,y,vx,vy)



