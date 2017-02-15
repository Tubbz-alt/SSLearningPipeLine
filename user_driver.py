from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np  

import psana

from sslearnpipeline import SSLearnPipeline

def get_dark(expname, run_number, output_filename, num_to_average=300):
    if not os.path.exists(output_filename):
        ds = psana.DataSource("exp=%s:run=%s:idx" % (expname, run_number))
        detector = psana.Detector('xtcav')
        run = ds.runs().next()
        event_times = list(run.times())
        random.seed(458921)
        random.shuffle(event_times)
        avg = None
        num_in_average = 0
        while num_in_average < num_to_average and len(event_times):
            tm = event_times.pop()
            evt = run.event(tm)
            img = detector.raw(evt)
            if img is None:
                continue
            img = img.astype(np.float32)
            if avg is None:
                avg = img
            else:
                weight = (num_in_average/float(num_in_average+1))
                avg *= weight
                avg += (1.0-weight)*img
            num_in_average += 1
        fout = file(output_filename,'w')
        np.save(fout, avg)
        fout.close()

    return np.load(file(output_filename,'r'))
    

def calc_log_thresh(img, thresh):
    img = img.astype(np.float32, copy=True)
    replace = img >= thresh
    newval = np.log(1.0 + img[replace] - thresh)
    img[replace]=thresh + newval
    return img


def calc_vproj_roi(img, window_len=224):
    assert len(img.shape)==2
    assert img.shape[1] >= window_len
    vproj = np.mean(img, axis=0)
    assert len(vproj)==img.shape[1]
    cumsum = np.cumsum(vproj)
    sliding_cumsum_over_windowlen = cumsum.copy()
    sliding_cumsum_over_windowlen[window_len:] -= cumsum[0:img.shape[1]-window_len]
    right = max(window_len, np.argmax(sliding_cumsum_over_windowlen))
    left = right - window_len
    adjust = 0
    if left < 0:
        adjust = -left
    if right >= img.shape[1]:
        adjust = img.shape[1] - right
    left += adjust
    right += adjust
    return [left, right]


def prepare_image(img, dark, is_present_threshold=510000, log_thresh=300, window_len=224):
    logsum = np.sum(np.log(np.maximum(1.0, 1.0 + img.astype(np.float))))
    if logsum < is_present_threshold:
        return None
    img = img.astype(np.float32)-dark.astype(np.float32)
    log_img = calc_log_thresh(img, log_thresh)
    left, right = calc_vproj_roi(log_img, window_len=window_len)
    roi_img = log_img
    roi_img  =log_img[:,left:right]
    roi_img = np.flipud(roi_img) 
    return roi_img


def get_eventid_for_filename(evt):
    img_id = evt.get(psana.EventId)
    id_for_filename = 'run-%d_sec-%d_nano-%d_fid-%d' % (img_id.run(),
                                                        img_id.time()[0],
                                                        img_id.time()[1],
                                                        img_id.fiducials())
    return id_for_filename


def main():    
    sslearn = SSLearnPipeline(outputdir='/reg/d/psdm/amo/amo86815/scratch/davidsch',
                              output_prefix='amo86815',
                              vgg16_weights='/reg/d/ana01/temp/davidsch/mlearn/vgg16/vgg16_weights.npz',
                              max_boxes_in_one_image=2,
                              total_to_label=7)
    dark_filename = '/reg/d/psdm/amo/amo86815/scratch/davidsch/dark_run68.npy'
    dark = get_dark(expname='amo86815', run_number=68, output_filename=dark_filename)
    
    ds = psana.DataSource('exp=amo86815:run=72:idx')
    detector = psana.Detector('xtcav')
    run = ds.runs().next()
    event_times = run.times()
    np.random.seed(394901)
    event_order = np.arange(len(event_times))
    np.random.shuffle(event_order)
    ii = 0
    while sslearn.labeling_not_done() and ii < len(event_order):
        tm = event_times[event_order[ii]]
        ii += 1
        evt = run.event(tm)
        orig_img = detector.raw(evt)
        if orig_img is None:
            continue
        prep_img = prepare_image(orig_img, dark)
        if prep_img is None:
            continue
        sslearn.label(prep_img, get_eventid_for_filename(evt))

    sslearn.build_models()

    for ii, tm in enumerate(event_times[0:100]):       
        evt = run.event(tm)
        orig_img = detector.raw(evt)
        if orig_img is None:
            continue
        prep_img = prepare_image(orig_img, dark)
        if prep_img is None:
            continue
        prediction = sslearn.predict(prep_img)
        if prediction['failed']:
            print("event %5d: prediction failed" % ii)
            continue
        print("event %5d: category=%d confidence=%.2f" % (ii, 
                                                          prediction['category'],
                                                          prediction['category_confidence']))
        for box in range(2):
            if prediction['boxes'][box]:
                boxdict = prediction['boxes'][box]
                print("    box %2d: confidence=%.2f xmin=%.1f xmax=%.1f  ymin=%.1f ymax=%.1f" % 
                      (box, boxdict['confidence'],
                       boxdict['xmin'], boxdict['xmax'], boxdict['ymin'], boxdict['ymax']))

        
    

if __name__ == '__main__':
    main()

