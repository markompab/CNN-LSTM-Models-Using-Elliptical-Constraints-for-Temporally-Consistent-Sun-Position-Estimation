"""
Sliding frames
--------------

That module provides the SlidingFrameGenerator that is helpful
to get more sequence from one video file. The goal is to provide decayed
sequences for the same action.


"""
import os.path
import traceback
import cv2 as cv
import numpy as np
import tensorflow as tf
from typing import Iterable

from datagenerators.keras_vid_generator_aug_framex2_channels_last_slidingframe import VideoFrameGenerator

class SlidingFrameGenerator(VideoFrameGenerator):
    """
    SlidingFrameGenerator is useful to get several sequence of
    the same "action" by sliding the cursor of video. For example, with a
    video that have 60 frames using 30 frames per second, and if you want
    to pick 6 frames, the generator will return:

    - one sequence with frame ``[ 0,  5, 10, 15, 20, 25]``
    - then ``[ 1,  6, 11, 16, 21, 26])``
    - and so on to frame 30

    If you set `sequence_time` parameter, so the sequence will be reduce to
    the given time.

    params:

    - sequence_time: int seconds of the sequence to fetch, if None, the entire \
        vidoe time is used

    from VideoFrameGenerator:

    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that
        will be replaced by one of the class list
    """

    def __init__(self, *args, sequence_time: int = None, step=5, sequencerange=300, step_btn=10, **kwargs):
        super().__init__(no_epoch_at_init=True, *args, **kwargs)
        #self.sequence_time = sequence_time
        self.sequence_length = sequence_time
        self.sequencerange      = sequencerange
        self.step_btn  = step_btn
        self.step = step

        self.sample_count = 0
        self.vid_info = []
        self.__frame_cache = {}
        self.__init_length()
        self.on_epoch_end()

    def __init_length(self):
        count = 0
        print("Checking files to find possible sequences, please wait...")
        for filename in self.files:
            cap = cv.VideoCapture(filename)
            fps = cap.get(cv.CAP_PROP_FPS)
            frame_count = self.count_frames(cap, filename)
            cap.release()

            if self.sequence_length is not None:
                #seqtime = int(fps * self.sequence_length)
                seqtime =  self.sequence_length
            else:
                seqtime = int(frame_count)

            #stop_at = int(seqtime - self.nbframe)
            #top_at = int(seqtime - (self.nbframe*self.frame_interval))

            if(seqtime>frame_count):
                seqtime = frame_count-1

            stop_at = int(seqtime - self.step_btn)
            step = self.step
            #step = np.ceil(seqtime / self.nbframe).astype(np.int) - 1
            i = 0

            #while stop_at > 0 and i <= frame_count - stop_at: # modified condition to ignore short video
            while stop_at > 0 and i < frame_count-(step*self.nbframe): # modified condition to ignore short video
                self.vid_info.append(
                    {
                        "id": count,
                        "name": filename,
                        "frame_count": int(frame_count),
                        #"frames": np.arange(i, i + stop_at)[::step][: self.nbframe],
                        "frames": np.arange(i, i + stop_at)[::step][: self.nbframe],
                        "fps": fps,
                    }
                )
                count += 1
                #i += 1
                i += self.step_btn

        nb = len(self.vid_info)
        print(
            "For %d files, I found %d possible sequence samples"
            % (self.files_count, len(self.vid_info))
        )
        #self.indexes = np.arange(len(self.vid_info))
        self.indexes = np.arange(len(self.vid_info))

    def on_epoch_end(self):
        # prepare transformation to avoid __getitem__ to reinitialize them
        if self.transformation is not None:
            self._random_trans = []
            for _ in range(len(self.vid_info)):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            #np.random.shuffle(self.indexes)
            tf.random.shuffle(self.indexes)

    def __len__(self):
        #return int(np.floor(len(self.vid_info) / self.batch_size))
        return int(tf.floor(len(self.vid_info) / self.batch_size))

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            labels_all=self.labels_all
            , use_frame_cache=True
            , sequence_time=self.sequence_length
            , nb_frames=self.nbframe
            , nb_channel=self.nb_channel
            , target_shape=self.target_shape
            , classes=self.classes
            , batch_size=self.batch_size
            , shuffle=self.shuffle
            , rescale=self.rescale
            , glob_pattern=self.glob_pattern
            , _validation_data=self.validation
            , sequencerange = self.sequencerange
            , step_btn = self.step_btn
            , step=self.step
        )

    def get_test_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            labels_all=self.labels_all
            , use_frame_cache=True
            , sequence_time=self.sequence_length
            , nb_frames=self.nbframe
            , nb_channel=self.nb_channel
            , target_shape=self.target_shape
            , classes=self.classes
            , batch_size=self.batch_size
            , shuffle=self.shuffle
            , rescale=self.rescale
            , glob_pattern=self.glob_pattern
            , _test_data=self.test
            , sequencerange = self.sequencerange
            , step_btn = self.step_btn
            , step=self.step

        )

    def extLabel(self, filters):
        dfilters = filters * 2
        #extfilters = np.repeat(dfilters, 2)
        extfilters = tf.repeat(dfilters, 2)
        #d = extfilters.copy()
        d = tf.Variable(extfilters)
        c =  tf.compat.v1.assign(d[1::2], d[1::2] + 1)

        # dfilters = filters * 2
        # extfilters = np.repeat(dfilters, 2)
        # d = extfilters.copy()
        # d[::2] = d[::2] - 1
        return c

    def __getitem0__(self, idx):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        transformation = None

        self.vidpaths  = []
        cvidpaths = self.vid_info

        for i in indexes:

            try:
                # prepare a transformation if provided
                if self.transformation is not None:
                    transformation = self._random_trans[i]

                vid = self.vid_info[i]
                video = vid.get("name")
                #classname = self._get_classname(video)

                # create a label array and set 1 to the right column
                #label = np.zeros(len(classes))
                #col = classes.index(classname)
                #label[col] = 1.0

                frame_label = self._get_label(video)

                video_id = vid["id"]

                if video_id not in self.__frame_cache:
                    frames: Iterable = self._get_frames(video, nbframe, shape)
                else:
                    frames: Iterable = self.__frame_cache[video_id]

                '''extract selected frames  '''
                frame_order = vid.get("frames")
                frames_chosen  = frames[frame_order]

                # apply transformation
                if transformation is not None:
                    frames_chosen = [
                        self.transformation.apply_transform(frame, transformation)
                        for frame in frames_chosen
                    ]


                frame_order_lbl = self.extLabel(frame_order)
                max = np.max(frame_order_lbl)
                #print("Frame Label:{}, max:{}".format(len(frame_label), max))

                if max >= len(frame_label):
                    continue

                label_vid   = frame_label[frame_order_lbl]

                # add the sequence in batch
                images.append(frames_chosen)
                labels.append(label_vid)
            except Exception as e :
                print(traceback.format_exc())

        # npimages = np.asarray(images).astype(np.float32)
        # nplabels = np.asarray(labels).astype(np.float32)

        #tf.print("images", len(images))

        # tfimages = tf.convert_to_tensor(npimages)
        # tflabels = tf.convert_to_tensor(nplabels)

        return np.array(images), np.array(labels)

    def __getitem__(self, idx):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        transformation = None

        self.vidpaths  = []
        cvidpaths = self.vid_info

        for i in indexes:

            try:
                # prepare a transformation if provided
                if self.transformation is not None:
                    transformation = self._random_trans[i]

                vid = self.vid_info[i]
                video = vid.get("name")
                #classname = self._get_classname(video)

                # create a label array and set 1 to the right column
                #label = np.zeros(len(classes))
                #col = classes.index(classname)
                #label[col] = 1.0

                frame_label = self._get_label(video)

                '''extract selected frames  '''
                frame_order = vid.get("frames")
                #frames_chosen = frames[frame_order]

                video_id = vid["id"]
                if video_id not in self.__frame_cache:
                    frames: Iterable = self._get_frames(video, nbframe, shape, frame_inds=frame_order)
                else:
                    frames: Iterable = self.__frame_cache[video_id]

                # apply transformation
                if transformation is not None:
                    frames = [
                        self.transformation.apply_transform(frame, transformation)
                        for frame in frames
                    ]

                frame_order_lbl = self.extLabel(frame_order)
                max = np.max(frame_order_lbl)
                #print("Count Frame:{} Label :{}, {} max:{}".format(len(frame_order),len(frame_label),frame_order_lbl, max))

                if max >= len(frame_label):
                    continue

                label_vid   = frame_label[frame_order_lbl]

                # add the sequence in batch
                images.append(frames)
                labels.append(label_vid)
            except Exception as e :
                print(traceback.format_exc())

        # npimages = np.asarray(images).astype(np.float32)
        # nplabels = np.asarray(labels).astype(np.float32)

        #tf.print("images", len(images))

        # tfimages = tf.convert_to_tensor(npimages)
        # tflabels = tf.convert_to_tensor(nplabels)
        a,b  = np.array(images), np.array(labels)
        #print("images:{} labels:{}".format(a.shape, b.shape))
        #return np.array(images), np.array(labels)
        return a, b

    def __getitemold__(self, idx):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        transformation = None

        self.vidpaths  = []
        cvidpaths = self.vid_info

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            vid = self.vid_info[i]
            video = vid.get("name")
            #classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            #label = np.zeros(len(classes))
            #col = classes.index(classname)
            #label[col] = 1.0

            label = self._get_label(video)

            video_id = vid["id"]
            if video_id not in self.__frame_cache:
                frames: Iterable = self._get_frames(video, nbframe, shape)
            else:
                frames: Iterable = self.__frame_cache[video_id]

            # apply transformation
            if transformation is not None:
                frames = [
                    self.transformation.apply_transform(frame, transformation)
                    for frame in frames
                ]

            '''extractslected frames  '''
            vids, lbls, cvidpaths = self.genFrameCombos(frames, label, video, cvidpaths)

            # add the sequence in batch
            # images.append(frames)
            # labels.append(label)
            images.append(vids)
            labels.append(lbls)

        #return np.array(images), np.array(labels)
        return tf.constant(images), tf.constant(labels)


    def genFrameCombos(self, orgframes, orglabels, vidname, vidinfo):
        '''extract vidinfo rows from orgframes'''
        vids = []

        cvidinfo = vidinfo
        lbls = []

        n = len(orgframes)
        for i in range(n):
            if(cvidinfo[i]['name'] == vidname):
                framelist = vidinfo[i]['frames']
                framesSelected = orgframes[framelist]

                vids.append(framesSelected)

                lblsSelected = orglabels[framelist]
                lbls.append(lblsSelected)

                cvidinfo.pop(i)

        return vids, lbls, cvidinfo