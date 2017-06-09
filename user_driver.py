from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import sys
import os
import random
import numpy as np  
import glob
import h5py
import xml.etree.ElementTree as ET
import psana
import scipy.misc
from sslearnpipeline import SSLearnPipeline





def prep_for_vgg(img, channel_mean):
    img2 = scipy.misc.imresize(img.astype(np.float32), (224, 224)) #, mode='F')
    img2 = img2.astype(np.float32)
    assert img2.dtype == np.float32
    img2 -= channel_mean
    img2 = np.reshape(img2, (224, 224, 1 ))
    img2 = np.repeat(img2, 3, axis=2)
    batch = np.reshape(img2, (1,224,224,3))
    return batch



def get_eventid_for_filename(evt):
    img_id = evt.get(psana.EventId)
    id_for_filename = 'run-%d_sec-%d_nano-%d_fid-%d' % (img_id.run(),
                                                        img_id.time()[0],
                                                        img_id.time()[1],
                                                        img_id.fiducials())
    return id_for_filename


def main():
    #put in directory /reg/g/psdm/tutorials/transferlearning
    sslearn = SSLearnPipeline(outputdir='/reg/g/psdm/tutorials/transferLearning',
                              output_prefix='amo86815',
                              vgg16_weights='/reg/g/psdm/tutorials/transferLearning/vgg16_weights.npz',
                              max_boxes_in_one_image=2,
                              total_to_label=60)

    ds = psana.DataSource("exp=amo86815:run=71:idx")
    detector = psana.Detector('xtcav')

    ds_run = ds.runs().next()
    ds_run_times = ds_run.times()

    old_dark = h5py.File('/reg/g/psdm/tutorials/transferLearning/xtcav_dark.h5','r')['dark'][:]


    arr = []
    A = np.load('indexlist.npy')


    for idx in A:
        #break
        index = idx

        #pull corresponding time and event number to get appropriate image

        tm = ds_run_times[index]
        evt = ds_run.event(tm)
        a = get_eventid_for_filename(evt)

        #all of these steps where we set datatypes are still imporant. the finals codewords can be quite sensitive to these types of things
        orig_img = detector.raw(evt).astype(np.float32)

        #subtract the dark

        img = (orig_img - old_dark).astype(np.int16)

        #Do appropriate resizing and image processing to make image suitable to pass through vggnet

        img = scipy.misc.imresize(img.astype(np.float), 50).astype(np.int16)
        img_prep = prep_for_vgg(img, channel_mean=8.46261599735)

        #pass through vggnet

        layers = sslearn.vgg16.get_model_layers(sslearn.session,imgs=img_prep, layer_names=['fc2'])
        codeword2 = layers[0][0,:]
        img = img_prep
        print(img.shape)
        sslearn.label(img, get_eventid_for_filename(evt))



    sslearn.build_models()

    #for ii, tm in enumerate(event_times[0:100]):       
    #    evt = run.event(tm)
    #    orig_img = detector.raw(evt)
    #    if orig_img is None:
    #        continue
    #    prep_img = prepare_image(orig_img, dark)
    #    if prep_img is None:
    #        continue
    #    prep_img -= ch_mean
    #    prediction = sslearn.predict(prep_img)
    #    if prediction['failed']:
    #        print("event %5d: prediction failed" % ii)
    #        continue
    #    print("event %5d: category=%d confidence=%.2f" % (ii, 
    #                                                      prediction['category'],
    #                                                      prediction['category_confidence']))
    #    for box in range(2):
    #        if prediction['boxes'][box]:
    #            boxdict = prediction['boxes'][box]
    #            print("    box %2d: confidence=%.2f xmin=%.1f xmax=%.1f  ymin=%.1f ymax=%.1f" % 
    #                  (box, boxdict['confidence'],
    #                   boxdict['xmin'], boxdict['xmax'], boxdict['ymin'], boxdict['ymax']))

    

if __name__ == '__main__':
    main()

