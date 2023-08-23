"""
Utils module to provide some nice functions to develop with Keras and
Video sequences.
"""

import random as rnd
import secrets

import matplotlib.pyplot as plt
import numpy as np


def show_sample(g, index=0, random=False, row_width=22, row_height=5):
    """ Displays a batch using matplotlib.

    params:

    - g: keras video generator
    - index: integer index of batch to see (overriden if random is True)
    - random: boolean, if True, take a random batch from the generator
    - row_width: integer to give the figure height
    - row_height: integer that represents one line of image, it is multiplied by \
    the number of sample in batch.
    """
    #total = len(g)
    total = g.files_count
    print(total)
    if random:
        sample = secrets.randbelow(total)
    else:
        sample = index

    #assert index < len(g)
    assert index < total
    print(g)
    #sample = g[sample]
    itm = []

    for i,row in enumerate(g):
        if(i==index):
            itm = row
            break
    sample = g.vid_info[sample]
    # sequences = sample[0]
    sequences = g.vid_info[0]
    #labels = sample[1]

    rows = len(sequences)
    index = 1
    plt.figure(figsize=(row_width, row_height * rows))
    for batchid, sequence in enumerate(sequences):
        #classid = np.argmax(labels[batchid])
        #classname = g.classes[classid]
        cols = len(sequence)
        for image in sequence:
            plt.subplot(rows, cols, index)
            #plt.title(classname)
            plt.imshow(image)
            plt.axis("off")
            index += 1
        #print(batchid, classname)
    plt.show()

def show_sample2(g, index=0, random=False, row_width=22, row_height=5):
    """ Displays a batch using matplotlib.

    params:

    - g: keras video generator
    - index: integer index of batch to see (overriden if random is True)
    - random: boolean, if True, take a random batch from the generator
    - row_width: integer to give the figure height
    - row_height: integer that represents one line of image, it is multiplied by \
    the number of sample in batch.
    """
    # total = len(g)
    total = g.files_count
    print(total)
    if random:
        sample = secrets.randbelow(total)
    else:
        sample = index

    # assert index < len(g)
    assert index < total
    print(g)
    # sample = g[sample]
    sample = g.vid_info[sample]
    # sequences = sample[0]
    sequences = g.vid_info[0]
    labels = sample[1]

    rows = len(sequences)
    index = 1
    plt.figure(figsize=(row_width, row_height * rows))
    for batchid, sequence in enumerate(sequences):
        classid = np.argmax(labels[batchid])
        # classname = g.classes[classid]
        cols = len(sequence)
        for image in sequence:
            plt.subplot(rows, cols, index)
            # plt.title(classname)
            plt.imshow(image)
            plt.axis("off")
            index += 1
        # print(batchid, classname)
    plt.show()