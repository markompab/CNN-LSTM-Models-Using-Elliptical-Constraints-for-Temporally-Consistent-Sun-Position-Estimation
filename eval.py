import getopt
import  os
import csv
import sys

import cv2
import math
import traceback
import numpy as np
import tensorflow as tf
import sklearn.metrics
from datetime import datetime
from keras.utils import losses_utils
from keras_preprocessing import image
from matplotlib import pyplot as plt
from utils.utils_data import UtilsData
from datagenerators.keras_vidgenerator_slidingframe import SlidingFrameGenerator

def writeToFile(path, content):
    f = open(path, "a")
    f.write(content)
    f.close()

def loadImg(impath, w, h):
    img = image.load_img(impath, target_size=(w, h))
    x = image.img_to_array(img)  # [:,:,0]
    #x = image.img_to_array(img)[:, :, 0]
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    return images
'''
    load data
'''

def cdateTm():
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")
    return date_time

def saveToFile(path, str):
    f = open(path, 'a')
    f.write(str)
    f.close()

def getTestGenerator(trainval_rows, STEP, STEP_BTN, SQRANGE):
    classes = [0, 1]
    return SlidingFrameGenerator(
        labels_all=trainval_rows
        , classes=classes
        , glob_pattern=glob_pattern
        , nb_frames=NBFRAME
        , split_test=.99
        , split_val=0.01#.3,
        # split=.33,
        , shuffle=False
        , batch_size=BS
        , target_shape=SIZE
        , nb_channel=CHANNELS
        # transformation=data_aug,
        , use_frame_cache=True
        , step=STEP
        , step_btn=STEP_BTN
        , sequencerange=SQRANGE
    )
    classes = [0, 1]

def pltDistScatter(savepath, gt, pred):

    fig = plt.figure(figsize=(10, 10))
    #fig.set_figwidth(12)

    ax = fig.add_subplot(111)
    x = pred[:, 1]
    y = pred[:, 0]
    #pred[:, 0] = 1 - pred[:, 0]
    #gt[:, 0] = 1 - gt[:, 0]

    ax.scatter(pred[:, 1], pred[:, 0], c='red', label="Predictions",s=1)
    ax.scatter(gt[:, 1], gt[:, 0], c='green', label="Ground truth", s=1)

    plt.ylabel("Elevation")
    plt.xlabel("Azimuth")
    plt.legend()
    #plt.show()

    plt.savefig(savepath)
    plt.clf()

def computeErrors(savepath, gt, pred):
    mae = sklearn.metrics.mean_absolute_error(gt, pred)
    mse = sklearn.metrics.mean_squared_error(gt, pred)
    rmse = sklearn.metrics.mean_squared_error(gt, pred, squared=False)
    var = sklearn.metrics.explained_variance_score(gt, pred)
    r2 = sklearn.metrics.r2_score(gt, pred)

    l = len(gt)

    str = "MAE:{} \nMSE:{} \nRMSE:{} \nVAR:{} \nR2 Square:{}\n Item Count:{}" \
        .format(mae, mse, rmse, var, r2, l)

    saveToFile(savepath, str)

def plotBbox(savepath, gt, pred):
    error = gt - pred
    error = np.power(error, 2)
    error = np.sqrt(np.sum(error, axis=-1))

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('RMSE', fontsize=14, fontweight='bold')
    fig.set_figwidth(3)

    ax = fig.add_subplot(111)
    ax.boxplot(error)
    plt.savefig(savepath)
    plt.clf()


def plotViolin(savepath, gt, pred):
    error = gt - pred
    error = np.power(error, 2)
    error = np.sqrt(np.sum(error, axis=-1))

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('RMSE', fontsize=14, fontweight='bold')
    fig.set_figwidth(3)

    ax = fig.add_subplot(111)
    ax.violinplot(error)
    plt.savefig(savepath)
    plt.clf()

def runOnAll():
    modelpath = "models/rgb"
    models = os.listdir(modelpath)

    data_path = "/home/cvlab/Learning/lavalsun/val_jpg/"
    csvpath = "/home/cvlab/Learning/lavalsun/val_exr/listed.csv"
    csvreader = csv.reader(open(csvpath))

    load_data = ""

    max_dists = [0.1, 0.2, 0.3, 0.4, 1]

    for i in range(len(models)):
        try:
            w,h = 500,500
            '''load model path '''
            model = tf.keras.models.load_model("{}/{}".format(modelpath, models[i]))
            #next(csvreader)

            for row in csvreader:
                try:
                    impath =  "{}{}jpg".format(data_path, row[0])
                    alt = float(row[1]) / (math.pi)
                    azi = 0.5 + float(row[2]) / (2 * math.pi)

                    gt = [alt, azi]

                    im = loadImg(impath, w, h)
                    pred = np.squeeze(model.predict(im))

                    error = np.absolute(pred - gt)
                    pp = pred[error < max_dists[i], :]
                    tp = gt[error < max_dists[i], :]

                    var = sklearn.metrics.explained_variance_score(tp, pp)
                    mae = sklearn.metrics.mean_absolute_error(tp, pp)
                    mse = sklearn.metrics.mean_squared_error(tp, pp)
                    rmse = sklearn.metrics.mean_squared_error(tp, pp, squared=False)

                    summ = "{}-{}\t\t variance: {}\t\t MAE: {}\t\t  MSE: {}\t\t  RMSE: {} imsize:{}" .format(models[i], max_dists[i], var, mae, mse, rmse, "{}x{}".format(h,w))

                    print(summ)
                    writeToFile("../stats/rgb.csv", summ)
                except Exception as e:
                    print("in")
                    print(traceback.format_exc())

        except Exception:
            print("out")
            print(traceback.format_exc())

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

def custom_loss_function(i):
   def loss(y_true, y_pred):
      print("pred:{} labels:{}".format(y_pred.shape, y_true.shape))
      h = tf.keras.losses.Huber()
      #huberloss = h([y_true], y_pred)
      wloss = wing_loss([y_true], y_pred)

      kl = tf.keras.losses.KLDivergence(
          reduction = losses_utils.ReductionV2.AUTO, name='kl_divergence'
      )

      #kdloss = kl([y_true], y_pred)#.numpy()

      '''Calculate curve smoothness'''
      '''
      y_pred2 =  tf.squeeze(y_pred)
      x = y_pred2[1::2]
      y = y_pred2[0::2]

      dy_dx = 0
      with tf.GradientTape() as tape:
         dy_dx = tape.gradient(y,x)
         print(dy_dx)

      #dy_dx = np.sum(np.diff(y) / np.diff(x))'''

      '''ordering score'''
      #h_score  = get_horder_score(x)

      #squared_difference = tf.square([y_true] - y_pred)
      #return tf.reduce_mean(squared_difference, axis=-1)
      alpha = 1
      #final_loss =  alpha * wloss + (1-alpha) * dy_dx #+ h_score
      #final_loss =  alpha * wloss + (1-alpha) * kdloss #+ h_score
      return wloss

      #return huberloss
      #return final_loss
   return loss

def addText(im, txt, x, y, color):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    #org = (50, 50)
    org = (x, y)

    # fontScale
    fontScale = 0.4

    # Blue color in BGR
    #color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    im_out = cv2.putText(im, txt, org, font,  fontScale, color, thickness, cv2.LINE_AA)

    return im_out

def batchNorm2Degrees(bt_norm):
    bt_rad = bt_norm.copy()

    '''Convert to Radians'''
    #bt_rad[:, ::2] = bt_rad[:, ::2] * math.pi/2
    bt_rad[:, ::2]  = bt_rad[:, ::2] * math.pi/4
    bt_rad[:, 1::2] = bt_rad[:, 1::2] * math.pi * 2

    '''Convert to Degrees'''''
    bt_d = np.rad2deg(bt_rad)

    return bt_d
def runOn4(modelPath, w, h, tlabel, test):


        '''load model path '''
        model = tf.keras.models.load_model(modelPath, custom_objects={'loss':custom_loss_function(1)})
        #next(csvreader)
        preds = None
        gts = None
        ts = len(test.files)
        test_vals = np.zeros((ts, NBFRAME * 2))
        #n = int(len(test.files)/BS)
        n = int(len(test.vid_info)/BS)

        hard_cases = []
        cnt  = 0
        for batch in test:
            '''predict'''
            gt   = batch[1]
            pred = model.predict(batch[0])

            gt0, pred0   = gt.copy(), pred.copy()
            # gt0[:,::2]   = 1 - gt0[:,::2]
            # pred0[:,::2] = 1 - pred0[:,::2]

            '''convert to degrees'''
            dgt   = batchNorm2Degrees(gt)
            dpred = batchNorm2Degrees(pred)

            '''append'''
            if(preds is None):
                preds = dpred[:, -2:]
                gts   = dgt[:,-2:]

            else:
                preds = np.concatenate((np.array(preds), dpred[:, -2:]), axis=0)
                gts   = np.concatenate((np.array(gts), dgt[:, -2:]), axis=0)

            '''Preview'''
            for i in range(BS):
                for j in range(NBFRAME):
                    im = batch[0][i,j]
                    gt_y = int(SIZE[1] * gt[i,j*2])
                    gt_x = int(SIZE[0] * gt[i,j*2+1])

                    pred_y = int(SIZE[1] * pred[i,j*2])
                    pred_x = int(SIZE[0] * pred[i,j*2+1])

                    r = 3
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                    # gt_dy, gt_dx = int(180 * gt[i,j*2]), int(360 * gt[i,j*2+1])
                    # pred_dy, pred_dx = int(180 * pred[i,j*2]), int(360 * pred[i,j*2+1])

                    gt_dy, gt_dx = dgt[i,j*2], dgt[i,j*2+1]
                    pred_dy, pred_dx = dpred[i,j*2], dpred[i,j*2+1]

                    log = "Gt: {:.2f} {:.2f} Pred: {:.2f} {:.2f}".format(gt_dx, gt_dy, pred_dx, pred_dy)
                    im = addText(im, log, 10, 20, (100, 149, 237))

                    dist = np.sqrt((gt_dx - pred_dx)**2 + (gt_dy - pred_dy)**2)
                    im = addText(im, "Dist: {}".format(dist), 10, 40, (100, 149, 237))

                    cv2.circle(im, (gt_x, gt_y), r, (255, 0, 0), -1)
                    cv2.circle(im, (pred_x, pred_y), r, (0, 0, 255), -1)

                    cv2.imshow("Frame", im)
                    cv2.waitKey(1)

                log = "{}/{}".format(cnt, n)
                print(log)

            cnt += 1
            if(cnt>=n):
                break

        path = "../evals/{}_hard cases.txt".format(cdateTm())
        np.savetxt(path, hard_cases,fmt="%s",)

        #test_vals = np.squeeze(test_vals)
        #print(test_vals.shape)
        #print(tlabel)

        #runEval(modelPath, np.array(preds), np.array(gts), h,w)
        dims = "{}x{}".format(h,w)
        gts =  np.squeeze(np.array(gts))
        preds = np.squeeze(np.array(preds))

        savepath  =  "../evals/{}_{}__{}_errors.txt".format(cdateTm(), tlabel, dims)
        computeErrors(savepath, gts, preds)

        savepath  =  "../evals/{}_{}_{}_scatter.jpg".format(cdateTm(), tlabel, dims)
        pltDistScatter(savepath, gts, preds)

        savepath  =  "../evals/{}_{}_{}_bbox.jpg".format(cdateTm(), tlabel, dims)
        plotBbox(savepath, gts, preds)

        savepath  =  "../evals/{}_{}_{}_bviolin.jpg".format(cdateTm(), tlabel, dims)
        plotViolin(savepath, gts, preds)

def runMetric(tp, pp, modelpath, max_dist):
    var = sklearn.metrics.explained_variance_score(tp, pp)
    mae = sklearn.metrics.mean_absolute_error(tp, pp)
    mse = sklearn.metrics.mean_squared_error(tp, pp)
    rmse = sklearn.metrics.mean_squared_error(tp, pp, squared=False)
    r2score = sklearn.metrics.r2_score(tp, pp)

    summ = "{}-{}\t\t variance: {}\t\t MAE: {}\t\t  MSE: {}\t\t  RMSE: {} \t\tR2 Score:{} \t\timsize:{}\n".format(modelpath, max_dist,
                                                                                            var, mae, mse, rmse,r2score,
                                                                                            "{}x{}".format(h,w))
    print(summ)
    writeToFile("../stats/rgb.csv", summ)


def setGPU(index):

    if(index == -1):
        tf.config.experimental.set_visible_devices([], 'GPU')
        return

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[index], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)


gpu = -1
setGPU(gpu)
print("GPU:{}".format(gpu))
classes = [0,1]

arglst = sys.argv[1:]
arg_options_short = "c:n:b:t:d:s:q:h:w:"
args_optionslong = [
                 "channels="
                , "nbframe="
                , "bs="
                , "step="
                , "step_btn="
                , "sqrange="
                , "height="
                , "width="
                , "test_csvpath="
                , "test_datapath="
                , "model_path="
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

SIZE = args['width'], args['height']
CHANNELS, NBFRAME, BS = args["channels"], args["nbframe"], args["bs"]
STEP, STEP_BTN, SQRANGE  = args["step"], args["step_btn"], args["sqrange"]
label = "resnet_lstm_{}_{}_sirtaraw_ecp_esp_stp_{}_stp_btn_{}_3lstm".format(NBFRAME, SIZE, STEP, STEP_BTN)


modelpath = None
if(args["model_path"] != ""):
    modelpath = args["model_path"]

trainval_rows = UtilsData.loadCSVDataExt(args["train_csvpath"], args["train_datapath"], "avi")
glob_pattern = args["train_datapath"]
print(len(trainval_rows))
train = getTestGenerator(trainval_rows, STEP, STEP_BTN, SQRANGE)
test = train.get_test_generator()
w, h = SIZE

runOn4(modelpath, w, h, label, test)
