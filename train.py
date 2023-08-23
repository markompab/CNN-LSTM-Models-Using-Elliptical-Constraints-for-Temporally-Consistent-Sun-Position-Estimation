import tensorflow as tf
import traceback
import getopt
import math
import sys
import os


from datagenerators.keras_vidgenerator_slidingframe import SlidingFrameGenerator
from tensorflow.python.ops.numpy_ops import np_config
from utils.newtonEllipsetf import NeutonEllipse
from utils.fitellipsetf import EllipseModel
from keras.callbacks import ModelCheckpoint
from utils.HistLogger import HistLogger
from utils.utils_time import UtilsTime
from utils.utils_data import UtilsData
from tensorflow import keras
from datetime import datetime

np_config.enable_numpy_behavior()

gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu) # second gpu

def genCNNLSTMModel(frames, channels, pixels_x, pixels_y, categories):
    # input_shape= (500,500,3)
    # input_shape= (1024,256,3)
    #input_shape = (1024, 256, 3)
    #input_shape = (128, 512, 3)
    #input_shape = (64, 256, 3)
    input_shape = (SIZE[1], SIZE[0], 3)

    #input_shape = (256, 1024, 3)
    base = tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False)
    x = keras.layers.Flatten()(base.output)
    #x = keras.layers.Dense(32, activation='linear')(x)
    x = keras.layers.Dense(NUM_OF_LANDMARKS, activation='linear')(x)
    res_model = tf.keras.Model(inputs=base.inputs, outputs=x)

    model = keras.Sequential()
    model.add(keras.layers.TimeDistributed(res_model))
    model.add(keras.layers.LSTM(16
                                ,kernel_initializer ="glorot_normal"
                                ,recurrent_activation="sigmoid"
                                ,activation="tanh")) #G5 #64 #32 #x161

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(NBFRAME * 2
                                #  , activity_regularizer = regularizers.L2(1e-5)
                                  , activation='leaky_relu'))

    return model

def getDataGenerator(trainval_rows, split_test=0.75, split_val=0.25, step=10, step_btn=10, sqrange=300):
    classes = [0, 1]
    return SlidingFrameGenerator(
        labels_all=trainval_rows
        ,classes=classes
        ,glob_pattern=glob_pattern
        ,nb_frames=NBFRAME
        , split_test=split_test
        , split_val=split_val
        #.32
        # split=.33
        ,shuffle=True
        ,batch_size=BS
        ,target_shape=SIZE
        ,nb_channel=CHANNELS
        # transformation=data_aug,
        ,use_frame_cache=True
        #,use_frame_cache=False
        #,sequence_time=.2
        #, sequence_time=NBFRAME
        ,step=step
        ,sequencerange=sqrange
        ,step_btn=step_btn
    )


total_dist = tf.Variable(0., dtype=tf.float64)
sumdistances = tf.Variable(0., dtype=tf.float64)
def ellipticShapePenalty(y_true, y_pred):


    #tf.print("Elliptic Shape Penalty start")
    v = len(y_true)

    '''Vid File'''
    for j in range(BS):
        try:
            ''''''
            ##ytrue  = tf.reshape(y_true[j],(NBFRAME, 2))*(180,360)
            ytrue  = tf.reshape(y_true[j],(NBFRAME, 2))
            #ypred  = tf.reshape(y_pred[j],(NBFRAME, 2))*(180,360)
            ypred = tf.reshape(y_pred[j],(NBFRAME, 2))

            ell3 = EllipseModel()
            a = ell3.estimate(ypred)
            xc1, yc1, a1, b1, theta1 = a[0], a[1], a[2], a[3], a[4]

            neutonEllipse = NeutonEllipse(xc1, yc1, a1, b1, theta1)
            sumdistances.assign(0.)

            for k in range(NBFRAME):
                #dist =  neutonEllipse.find_distance2(ypred[k, 0], ypred[k,1])["error"]
                x = tf.stack([ypred[k, 0], ypred[k,1]])
                #tf.print("point_x: ", x)
                dist = neutonEllipse.find_distance2( x, tolerance=1e-4, max_iterations=100)["error"]
                sumdistances.assign_add(dist)

            # if(sumdistances > 0):
            if(tf.greater(sumdistances, 0.)):
                meanDist = tf.math.divide(sumdistances,NBFRAME)
                total_dist.assign_add(meanDist)

        except Exception as e:
            print(traceback.format_exc())

    #average distance
    #total_dist = total_dist #/(BS

    #if(total_dist is None):
    if(tf.math.is_nan(total_dist)):
        total_dist.assign(0.)

    return total_dist

total_distecp = tf.Variable(0., dtype=tf.float64)
sumdistancesecp = tf.Variable(0., dtype=tf.float64)
def ellipticConsistencyPenalty(y_true, y_pred):
    #total_dist = 0.

    v = len(y_true)

    '''Vid File'''
    for j in range(BS):
        try:
            ''''''
            #ytrue  = tf.reshape(y_true[j],(NBFRAME, 2))*(180,360)
            ytrue  = tf.reshape(y_true[j],(NBFRAME, 2))
            #ypred  = tf.reshape(y_pred[j],(NBFRAME, 2))*(180,360)
            ypred  = tf.reshape(y_pred[j],(NBFRAME, 2))

            ell1 = EllipseModel()
            #xc1, yc1, a1, b1, theta1 = 1,1,1,1,1
            a =  ell1.estimate(ytrue)
            xc1, yc1, a1, b1, theta1 = a[0], a[1], a[2], a[3], a[4]

            ell2 = EllipseModel()
            #xc2, yc2, a2, b2, theta2 = 100,100,100,100,100
            b = ell2.estimate(ypred)
            xc2, yc2, a2, b2, theta2 = b[0], b[1], b[2], b[3], b[4]

            d = tf.stack([xc1, yc1]) - tf.stack([xc2, yc2])
            dist = tf.norm(d) + tf.math.abs(a2 - a1) + tf.math.abs(b2 - b1) + tf.math.abs(theta2 - theta1)
            #total_dist += dist
            total_distecp.assign_add(dist)

        except Exception as e:
            print(traceback.format_exc())

    #average distance
    #if v > 0:
        #total_dist = total_dist./(v)
    ttdf = tf.math.divide_no_nan(total_distecp,tf.cast(v, tf.float64))
    total_distecp.assign(ttdf)

    '''
    Normalise    =   center distance            w                  h                tilt 
    Max distance =  SQRT(h**2 + w**2)  + SQRT(h**2 + w**2) + SQRT(h**2 + w**2)   +   180
    '''
    #max_dist = math.sqrt(SIZE[0]**2 + SIZE[1]**2)
    max_dist = math.sqrt(SIZE[0]**2 + SIZE[1]**2)
    #ntotal_dist  = total_dist/max_dist
    ntotal_dist  = tf.math.divide_no_nan(total_distecp,max_dist)

    return ntotal_dist
def wing_loss(landmarks, labels, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('wing_loss'):
        #print("pred:{} labels:{}".format(landmarks.shape, labels.shape))
        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w/epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.math.log(1.0 + absolute_x/epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss

train_counter = 0
def custom_loss_function(i):
   def loss(y_true, y_pred):
      global train_counter
      train_counter += 1
      wloss = wing_loss([y_true], y_pred)
      ecp  = ellipticConsistencyPenalty(y_true, y_pred)
      esp  = ellipticShapePenalty(y_true, y_pred)

      '''ordering score'''
      alpha = 0.65
      lambda1 = (1-alpha) * 0.5
      lambda2 = 1 - (alpha+lambda1)

      with train_writer.as_default(step=train_counter):
          tf.summary.scalar('wloss', wloss)
          tf.summary.scalar('ecp', ecp)
          tf.summary.scalar('esp', esp)

      #final_loss =  alpha * wloss + (1-alpha) * ellipseloss #+ h_score
      final_loss =  alpha * wloss + lambda1 * ecp + lambda2 * esp #+ h_score
      #final_loss =  alpha * wloss + (1-alpha) * ecp  #+ h_score
      #final_loss =   wloss +  esp  + ecp#+ h_score

      #wandb.log({"alpha": alpha, "lambda1": lambda1, "lambda2": lambda2})
      #wandb.log({"wloss": wloss, "ecp": ecp, "esp": esp,  "final_loss": final_loss})


      #return final_loss
      return final_loss
   return loss

def tf_cond(x):
    x = tf.convert_to_tensor(x)
    s = tf.linalg.svd(x, compute_uv=False)
    r = s[..., 0] / s[..., -1]
    # Replace NaNs in r with infinite unless there were NaNs before
    x_nan = tf.reduce_any(tf.math.is_nan(x), axis=(-2, -1))
    r_nan = tf.math.is_nan(r)
    r_inf = tf.fill(tf.shape(r), tf.constant(math.inf, r.dtype))
    tf.where(x_nan, r, tf.where(r_nan, r_inf, r))
    return r

def trainCnnLSTM(log_label, modelPath=None, epochs=300):
    channels = 3
    pixels_x, pixels_y = SIZE  # 1024, 256
    categories = ["elevation", "azimuth"]
    ls_label = 'models/{}_resnet_lstm_{}.png'.format(UtilsTime.cdateTm(),log_label)
    md_label = 'models/{}_resnet_lstm_{}.h5'.format(UtilsTime.cdateTm(),log_label)
    ep_label = 'resnet_lstm_{}.h5'.format(log_label)

    model = None
    if(modelPath == None):
        model = genCNNLSTMModel(NBFRAME, channels, pixels_x, pixels_y, categories)

    else:
        model = keras.models.load_model(modelPath, custom_objects={'loss':custom_loss_function(1)})

    model.compile(optimizer='adam', loss=custom_loss_function(1))
    #model.compile(optimizer='adam', loss='mse')

    checkpointer = ModelCheckpoint(filepath=md_label, verbose=1, save_best_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
    #cb_epochvisuals = EpochLogger(ep_label, test, test_vals, NBFRAME)
    # history = model.fit_generator(
    history = model.fit(
        train  # generate_arrays(train_ids[:10])
        , validation_data=valid
        , epochs=epochs
        , batch_size=BS
        , verbose=1
        , shuffle=True
        #, initial_epoch=0
        , callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            , tb_callback
            , checkpointer
            #, cb_epochvisuals
        ]
    )
    HistLogger.saveHist(ls_label, history)

def testLosses(datagenerator, len, BS, NBFrame):

    i = 0
    for row in datagenerator:
        if (i == len):
            break
        try:


            row2 = row[1] +0.01

            a =ellipticShapePenalty(row[1], row2)
            #a = ellipticConsistencyPenalty(row[1], row2)

            print("Loss:{}".format(a))
        except Exception as e:
            print(traceback.format_exc())

        i = i + 1


NUM_OF_LANDMARKS =64

print("GPU:{}".format(gpu))
arg_options_short = "c:n:b:s:q:h:w:t:d:e:m:"

arglst = sys.argv[1:]
args_optionslong = [
                 "channels="
                ,  "nbframe="
                , "bs="
                , "step="
                , "step_btn="
                , "sqrange="
                , "height="
                , "width="
                , "trainval_split="
                , "train_datapath="
                , "train_csvpath="
                , "test_csvpath="
                , "test_datapath="
                , "model_path="
                , "epochs="
            ]
options, vals = getopt.getopt(arglst, arg_options_short, args_optionslong)
args = {"channels" : "3"
            , "nbframe" : "3"
            , "bs"      : "16"
            , "train_csvpath"  : ""
            , "train_datapath" : ""
            , "trainval_split" : "0.65"
            , "test_csvpath"   : ""
            , "test_datapath" : ""
            , "step" : "5"
            , "step_btn" : "1"
            , "sqrange": "300"
            , "height": "128"
            , "width": "512"
            , "model_path" : ""
            , "epochs" :  "300"                                                                                                                                                                                                            ""
}

for opt, arg in options:
    args[opt.replace("--", "")] = arg

'''parse'''
for argkey in args.keys():

    if("path" not in argkey and "split" not in argkey):
        args[argkey] = int(args[argkey])

    elif ("split" in argkey):
        args[argkey] = float(args[argkey])

print (args)

SIZE = args['width'], args['height']
CHANNELS, NBFRAME, BS = args["channels"], args["nbframe"], args["bs"]

for arg in sys.argv:
    print(arg)

train_log_dir = 'logs/gradient_tape/' + UtilsTime.cdateTm() + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

log_dir = 'logs/batch_level/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
train_writer = tf.summary.create_file_writer(log_dir)

trainval_rows = UtilsData.loadCSVDataExt(args["train_csvpath"], args["train_datapath"], "avi")
glob_pattern = args["train_datapath"]
print(len(trainval_rows))

print("STEP:{}, STEP_BTN:{}, SQRANGE:{}".format(args["step"], args["step_btn"], args["sqrange"]))

train = getDataGenerator(trainval_rows, split_test=.001, split_val=0.35, step=args["step"], step_btn=args["step_btn"], sqrange=args["sqrange"])
valid = train.get_validation_generator()

test_csvpath  = args["test_csvpath"]
test_datapath = args["test_datapath"]
glob_pattern2 = args["test_datapath"]

testval_rows = UtilsData.loadCSVDataExt(args["test_csvpath"], args["test_datapath"], "avi")
log_label = "{}_{}x{}_sirtaraw_alldatamc_ecp_esp_stp{}_stpbtn_{}".format( args["nbframe"],  args["height"], args["width"], args["step"], args["step_btn"])

modelpath = None
if(args["model_path"] != ""):
    modelpath = args["model_path"]

epochs = args["epochs"]

test = getDataGenerator(testval_rows, split_test=.01, split_val=0.001,  step=args["step"], step_btn=args["step_btn"], sqrange=args["sqrange"]).get_test_generator()
trainCnnLSTM(log_label, modelpath, epochs)
