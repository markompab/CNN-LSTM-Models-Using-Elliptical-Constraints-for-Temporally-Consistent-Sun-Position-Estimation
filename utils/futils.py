import csv
import glob
import os
import datetime
from os.path import isfile, join


class FUtils:
    @staticmethod
    def listFiles(source_dir, ext):
        files = glob.iglob(os.path.join(source_dir, "*.{}".format(ext)))
        return files

    @staticmethod
    def listAllFiles(source_dir):
        onlyfiles = [f for f in os.listdir(source_dir) if isfile(join(source_dir, f))]
        return onlyfiles

    @staticmethod
    def getParentFolderNm(path):
        dir = os.path.dirname(path)
        return os.path.basename(dir)


    @staticmethod
    def getFileNm(path):
        return os.path.basename(path)


    @staticmethod
    def loadCSVArray(path):
        results = []

        with open(path, 'r') as csvfile:
            #reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
            reader = csv.reader(csvfile)  # retain
            for row in reader:  # each row is a list
                results.append(row)

        return results

    @staticmethod
    def getCTime():
        now = datetime.datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")
