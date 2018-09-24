import os
import numpy as np
from PIL import Image

dataset = 'miniimagenet'
train_count = 10 # train samples per class
val_count = 590 # val samples per class

mnist_dir = '/home/prithvi/dsets/MNIST/trainingSet/'
omniglot_dir = '/home/prithvi/dsets/Omniglot/train'
miniimagenet_dir = '/home/prithvi/dsets/miniimagenet/train'

save_dir = './data'

if dataset == 'mnist':
    data_dir = mnist_dir
    size = (28,28)
elif dataset == 'omniglot':
    data_dir = omniglot_dir
    size = (28,28)
elif dataset == 'miniimagenet':
    data_dir = miniimagenet_dir
    size = (84,84)

if __name__ == '__main__':
    train_dir = os.path.join(save_dir,'metatrain')
    val_dir = os.path.join(save_dir,'metaval')

    if not (os.path.exists(train_dir)):
        os.system('mkdir '+train_dir)
    if not (os.path.exists(val_dir)):
        os.system('mkdir '+val_dir)

    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir,cls)
        
        cls_train_dir = os.path.join(train_dir,cls)
        if not os.path.exists(cls_train_dir):
            os.system('mkdir '+cls_train_dir)
        
        cls_val_dir = os.path.join(val_dir,cls)
        if not os.path.exists(cls_val_dir):
            os.system('mkdir '+cls_val_dir)
        
        samples = map(lambda x:(Image.open(x).resize(size,resample=Image.LANCZOS),
                                os.path.split(x)[-1]),
                        [os.path.join(cls_dir,s) for s in os.listdir(cls_dir)])
        np.random.shuffle(samples)
        train,val = samples[:train_count],samples[train_count:train_count+val_count]
        
        for s,fname in train:
            s.save(os.path.join(cls_train_dir,fname))
        for s,fname in val:
            s.save(os.path.join(cls_val_dir,fname))



