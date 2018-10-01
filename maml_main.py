"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import os
import csv
import sys
import numpy as np
import pickle
import random
import tensorflow as tf
from matplotlib import pyplot as plt

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'mnist', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 10, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('img_size',28,'image size')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 1000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 4, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 1, 'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_float('target_task_lr',1e-4,'step size for target task')

flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('target_maml_iterations',50,'number of MAML iterations at the end which use the target task class ordering')
flags.DEFINE_integer('target_task_iterations',1,'number of iterations on the target task after MAML pretraining')

flags.DEFINE_string('opt_mode','maml','mode for optimization: maml or plain sgd') # 'maml' or 'sgd'

## SGD options
flags.DEFINE_integer('sgd_bz',100,'batch size for plain sgd')
flags.DEFINE_integer('sgd_iterations',1000,'number of iterations for plain sgd')
flags.DEFINE_float('sgd_lr',1e-3,'learning rate for plain sgd')

## Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 32, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 5
    SAVE_INTERVAL = 100
        
    PRINT_INTERVAL = 10
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*50000

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        
        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            feed_dict = {}
            input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

def main(class_folders):
    test_num_updates = 1

    if FLAGS.metatrain_iterations == 0:
        print('ERROR: 0 metatrain itertions')
        return
    
    # list of class folders -- TODO: pass from experiment script
    #class_folders = sorted(['./data/metatrain/'+c for c in os.listdir('./data/metatrain')])
       
    data_generator = DataGenerator(FLAGS.update_batch_size*2+8, FLAGS.meta_batch_size, class_folders)
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    tf_data_load = True
    num_classes = data_generator.num_classes

    random.seed(5)
    image_tensor, label_tensor = data_generator.make_data_tensor()
    inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
    inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
    labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
    labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
    input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    print('>>>',inputa.get_shape().as_list(),'--',inputb.get_shape().as_list())

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    model.construct_model(input_tensors=input_tensors, prefix='metatrain_')

    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        #input('here??')
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    opt = FLAGS.opt_mode #'maml'
    
    # Plain SGD on target task
    print('Loading target task...')
    x,y,vx,vy = data_generator.load_target_task(class_folders)
    print('Target task shape:',x.shape,'--',y.shape,'__',vx.shape,'--',vy.shape)
    
    '''for i in range(3):
        plt.imshow(x[i].reshape(28,28,3))
        plt.title('Train '+str(y[i]))
        plt.show()
    for i in range(3):
        plt.imshow(vx[i].reshape(28,28,3))
        plt.title('Val '+str(vy[i]))
        plt.show()
    '''
    
    if opt == 'maml':
        print('Pretraining with MAML...')
        train(model, saver, sess, exp_string, data_generator, resume_itr)
        model.setup_maml_pred()
        #model.weights = model.fast_weights
        py,pvy = sess.run([model.maml_pred_x,model.maml_pred_vx],feed_dict={model.x_ph:x, model.y_ph:y, model.vx_ph:vx})
        py,pvy = py.argmax(axis=1),pvy.argmax(axis=1)
        #pvy = sess.run(model.maml_pred_vx,feed_dict={model.x_ph:vx}).argmax(axis=1)
    elif opt == 'sgd':
        #model.setup_final_pred(mode='sgd')
        print('Finetuning on the target task')
        bz = 100
        niter = 1000
        for i in range(niter):
            bi = np.random.randint(0,x.shape[0],bz)
            loss,_ = sess.run([model.sgd_loss,model.sgd_step],feed_dict={model.x_ph:x[bi],model.y_ph:y[bi]})
            if (i+1)%100 == 0:
                print('Iteration '+str(i)+' - '+str(np.mean(loss)))
        py = sess.run(model.sgd_pred,feed_dict={model.x_ph:x, model.y_ph:y}).argmax(axis=1)
        pvy = sess.run(model.sgd_pred,feed_dict={model.x_ph:vx}).argmax(axis=1)
    
    y,vy = y.argmax(axis=1),vy.argmax(axis=1)
    print('Train acc:',np.mean(py==y))
    print('Val acc:',np.mean(pvy==vy))

    #sess.close()

if __name__ == "__main__":
    dataset = 'mnist'
    metatrain_folder = './data/metatrain'
    if dataset == 'omniglot':
        pass
    elif dataset == 'miniimagenet':
        class_folders = sorted([os.path.join(metatrain_folder,c) for c in os.listdir(metatrain_folder) if c.startswith('n')])
    elif dataset == 'mnist':
        class_folders = sorted([os.path.join(metatrain_folder,c) for c in os.listdir(metatrain_folder) if (c[0] in map(str,range(10)))])
    
    # sample classes for N-way classification task
    class_folders = random.sample(class_folders,FLAGS.num_classes)
    
    # TODO: Sample K-shots from the training sets
    
    main(class_folders)


