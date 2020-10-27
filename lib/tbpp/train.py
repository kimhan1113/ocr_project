import keras

from ..datasets.ssd_data import InputGenerator
from .tbpp_training import TBPPFocalLoss
from ..utils.training import Logger

def train_tbpp(model,gt_util,prior_util,config,output_dir,freeze=None):

    epochs = config.TRAIN.NUM_EPOCHS 
    batch_size = config.TRAIN.BATCH_SIZE 

    gen_train = InputGenerator(gt_util, prior_util, batch_size, model.image_size,augmentation=True)
    # gen_val = InputGenerator(gt_util_val, prior_util, batch_size, model.image_size)

    for layer in model.layers:
        layer.trainable = not layer.name in freeze
    optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)

    # weight decay
    regularizer = keras.regularizers.l2(5e-4) # None if disabled
    regularizer = None
    for l in model.layers:
        if l.__class__.__name__.startswith('Conv'):
            l.kernel_regularizer = regularizer

    loss = TBPPFocalLoss()

    model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)


    history = model.fit_generator(
            gen_train.generate(),
            steps_per_epoch=int(gen_train.num_batches/4), 
            epochs=epochs, 
             verbose=1, 
             callbacks=[
                 keras.callbacks.ModelCheckpoint(output_dir+'/weights.{epoch:03d}.h5',
                                                 verbose=1, save_weights_only=True,period=10),
                 Logger(output_dir)
             ], 
             # validation_data=gen_val.generate(), 
             # validation_steps=gen_val.num_batches, 
             class_weight=None,
             max_queue_size=1, 
             workers=1, 
             use_multiprocessing=False, 
             initial_epoch=0 
            )

