import csv
import os
import traceback

import cv2
import numpy as np
from random import shuffle


class UtilsData:
    @staticmethod
    def loadCSVDataPlain(path):
        filecsv = open(path)
        csvreader = csv.reader(filecsv)
        rows = []
        header = []
        header = next(csvreader)

        for row in csvreader:
            try:

                new_row = {'elevation': float(row[1])  # *90,
                    , 'azimuth': float(row[2])  # *360,
                    , 'impath': row[0]}

                rows.append(new_row)

            except Exception as e:
                print(e)
        print("{}:{}".format(len(rows), len(rows[0])))
        return rows

    @staticmethod
    def loadCSVData(path, rootpath):
        filecsv = open(path)
        csvreader = csv.reader(filecsv)
        rows = []

        header = []
        header = next(csvreader)

        for row in csvreader:
            try:

                impath = "{}/{}.jpg".format(rootpath, row[0])

                new_row = {'elevation': float(row[1]),
                           'azimuth': float(row[2]),
                           'impath': impath}

                rows.append(new_row)

            except Exception as e:
                print(e)

        return rows

    @staticmethod
    def loadCSVDataExt(path, rootpath, ext):
        filecsv = open(path)
        csvreader = csv.reader(filecsv)
        rows = []

        header = []
        header = next(csvreader)

        for row in csvreader:
            try:

                impath = "{}/{}.{}".format(rootpath, row[0], ext)


                new_row = {'elevation': float(row[1]),
                           'azimuth': float(row[2]),
                           'impath': impath}

                rows.append(new_row)

            except Exception as e:
                print(e)

        return rows

    @staticmethod
    def formatMultiExpoCSV(srcdir, orgdata):

        m = len(orgdata)
        files = os.listdir(srcdir)
        n = len(files)

        frows = []
        for i in range(n):
            for j in range(m):
                if (orgdata[j]["impath"] in files[i]):
                    frow = orgdata[j].copy()
                    frow["impath"] = "{}/{}".format(srcdir, files[i])
                    frows.append(frow)
                    log = "{}/{}_{}: {}>>{}".format(i, n, j,orgdata[j]["impath"]  ,frow["impath"])
                    print(log)
                    break

        return frows

    @staticmethod
    def loadVideo(path):
        frames = []
        cap = cv2.VideoCapture(path)
        ret = True

        while (ret):
            ret, img = cap.read()
            if ret:
                # img = np.swapaxes(img, 0, 1)
                frames.append(np.transpose(img / 255))

        video = np.stack(frames, axis=0)

        return video

    @staticmethod
    def getLabels(test, NBFRAME, BS):
        ts = len(test.files)
        test_vals = np.zeros((ts, NBFRAME * 2))

        print(len(test.files))
        for n, file in enumerate(test.files):
            if (not os.path.exists(file) or n >= ts * BS):
                continue

            a =  test._get_label(file)
            if len(a) < 1:
                print("empty")

            test_vals[n] = a
            #test_vals[n] = test._get_label(file)

        test_vals = np.squeeze(test_vals)
        print(test_vals.shape)
        return test_vals

    @staticmethod
    def generate_arrays(rows):
        # print(rows)
        while True:

            shuffle(rows)
            for row in rows:
                try:
                    print(row["impath"])
                    scene = UtilsData.loadVideo(row["impath"])
                    category = np.array([row["elevation"], row["azimuth"]])
                    # print(scene.shape)
                    '''
                    scene =  cv2.VideoCapture(row[2])
                    category = (row[0],row[1])
                    '''
                    yield (np.array([scene]), category)
                except Exception:
                    print("fail-> {}".format(row["impath"]))
                    print(traceback.format_exc())

    @staticmethod
    def clean(rows):
        new_rows = []

        for row in rows:
            if (os.path.exists(row['impath'])):
                new_rows.append(row)

        return new_rows