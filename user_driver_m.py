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
#sys.path.append('/reg/neh/home/davidsch/mlearn/vgg_xtcav')

import vgg16img2codeword



h5 = h5py.File('/reg/d/ana01/temp/davidsch/mlearn/vgg_codewords_xtcavimg_small.h5','r')
fnames = h5['fname'][:]
rows = h5['row'][:]
lookup = {}
for idx,fname,row in zip(range(len(fnames)), fnames, rows):
    if row not in lookup:
        lookup[row]={}
    lookup[row][os.path.split(fname)[1]]=idx


def prep_for_vgg(img, channel_mean):
    img2 = scipy.misc.imresize(img.astype(np.float32), (224, 224)) #, mode='F')
    img2 = img2.astype(np.float32)
    assert img2.dtype == np.float32
    img2 -= channel_mean
    img2 = np.reshape(img2, (224, 224, 1 ))
    img2 = np.repeat(img2, 3, axis=2)
    batch = np.reshape(img2, (1,224,224,3))
    return batch


def get_dark(expname, run_number, output_filename, num_to_average=300):
    if not os.path.exists(output_filename):
        ds = psana.DataSource("exp=%s:run=%s:idx" % (expname, run_number))
        detector = psana.Detector('xtcav')
        run = ds.runs().next()
        event_times = list(run.times())
        random.seed(1046)
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
    
def assert_max_diff_less_than(A,B,max_diff_thresh,failmsg):
    diff = np.abs(A-B)
    max_diff = np.max(diff)
    assert max_diff < max_diff_thresh, failmsg

def report_diff(A,B,msg):
    diff = np.sum(np.abs(A-B))
    sum_all = 0.5 * np.sum(np.abs(A)+np.abs(B))
    percent_diff = diff/sum_all
    print("precent_diff=%.2f%% %s" % (100.0 * percent_diff, msg))


def assert_equal(A,B, failmsg):
    diff = np.abs(A-B)
    sum_diff = np.sum(diff)
    assert 0 == sum_diff, failmsg

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


#def parseLabelMeXml(fname, verbose=False):
#    tree = ET.parse(fname)
#    root = tree.getroot()
#    assert root.tag == 'annotation'
#    localizations = {'e1':None, 'e2':None}

#    for obj in root.iterfind('object'):
#        name = obj.find('name').text
#        if name == 'e3':
#            name = 'e2'
#        assert 'bounding_box' == obj.find('type').text, "unexpected, fname=%s, #object with type=%s, not bounding_box" % \
#            (fname, obj.find('type').text)
#        poly = obj.find('polygon')
#        pts = poly.findall('pt')
#        xs = [int(pt.find('x').text) for pt in pts]
#        ys = [int(pt.find('y').text) for pt in pts]
#        xmin, xmax = min(xs), max(xs)
#        ymin, ymax = min(ys), max(ys)
#        assert name in localizations, "name=%s" % name
#        assert localizations[name] is None
#        localizations[name]={'x':(xmin,xmax),
#                             'y':(ymin,ymax)}
#    return localizations


#def get_old_info(fname, verbose=False):
#    stem = os.path.splitext(os.path.basename(fname))[0]
#    h5_fname_part, row = stem.split('-row')
#    row = int(row)
#    h5_fname = h5_fname_part.split('-en')[0] + '.h5'
#    locdata = get_old_info_helper(fname)
#    localizations = parseLabelMeXml(fname, verbose)
#    locdata['e1'] = localizations['e1']
#    locdata['e2'] = localizations['e2']
#    locdata['h5fname'] = h5_fname
#    locdata['row'] = row
#    add_codewords(locdata)
#    return locdata

#def add_codewords(locdata):
#    global lookup
#    row = locdata['row']
#    fname = locdata['h5fname']
#    idx = lookup[row][fname]
#    locdata['codeword1']=h5['codeword1'][idx]
#    locdata['codeword2']=h5['codeword2'][idx]

#def get_old_info_helper(fname):
#    stem = os.path.splitext(os.path.basename(fname))[0]
#    xx,yy=stem.split('-en')
#    row = int(yy.split('-row')[1])
#    h5fname_small = os.path.join('/reg/d/ana01/temp/davidsch/ImgMLearnSmall', x#x + '.h5')
#    h5fname_large = os.path.join('/reg/d/ana01/temp/davidsch/ImgMLearnFull', xx# + '.h5')
#    assert os.path.exists(h5fname_small)
#    assert os.path.exists(h5fname_large)
#    h5_small = h5py.File(h5fname_small,'r')
#    h5_large = h5py.File(h5fname_large,'r')
#    res = {}
#    res['fiducials'] = h5_small['evt.fiducials'][row]
#    res['nanoseconds'] = h5_small['evt.nanoseconds'][row]
#    res['seconds'] = h5_small['evt.seconds'][row]
#    res['enPeaksLabel'] = h5_small['acq.enPeaksLabel'][row]
#    res['run'] = h5_small['run'][row]
#    res['run.index'] = h5_small['run.index'][row]
#    res['img_small'] = h5_small['xtcavimg'][row]
#    res['img_large'] = h5_large['xtcavimg'][row]
#    return res


def prepare_image(img, dark, is_present_threshold=510000, log_thresh=300, window_len=224):
    logsum = np.sum(np.log(np.maximum(1.0, 1.0 + img.astype(np.float))))
    if logsum < is_present_threshold:
        return None
    img = img.astype(np.float32)-dark.astype(np.float32)
    log_img = calc_log_thresh(img, log_thresh)
    log_img = np.maximum(0.0, log_img)
    log_img *= 254.0/350.0
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
    sslearn = SSLearnPipeline(outputdir='/reg/d/psdm/amo/amo86815/scratch/mmongia',
                              output_prefix='amo86815',
                              vgg16_weights='/reg/d/ana01/temp/davidsch/mlearn/vgg16/vgg16_weights.npz',
                              max_boxes_in_one_image=2,
                              total_to_label=60)
    dark_filename = '/reg/d/psdm/amo/amo86815/scratch/davidsch/dark_run68.npy'
    dark = get_dark(expname='amo86815', run_number=68, output_filename=dark_filename)
    

    #xmls = glob.glob('/reg/neh/home/davidsch/mlearn/localization/Annotations/users/goedelsch/amo86815/*.xml')
    ds = psana.DataSource("exp=amo86815:run=71:idx")
    detector = psana.Detector('xtcav')
    ds_run = ds.runs().next()
    ds_run_times = ds_run.times()
    sslearn_dark = np.load(file('/reg/d/psdm/amo/amo86815/scratch/davidsch/dark_run68.npy','r')).astype(np.float32)
    old_dark = h5py.File('/reg/neh/home/davidsch/rel/ImgMLearn/h5out/xtcav_dark.h5','r')['dark'][:]
    arr = []
    A = np.load('indexlist.npy')
    #for xml in xmls:
    #    locdata = get_old_info(xml)
    #    if locdata['run'] != 71: continue
    #    index = locdata['run.index']
    #    A.append(index)

    #np.save('indexlist',A)
    for idx in A:
        #break
        #locdata = get_old_info(xml)
        #print(locdata)
        #raw_input("hola")
        #if locdata['run'] != 71: continue
        index = idx
        tm = ds_run_times[index]
        evt = ds_run.event(tm)
        a = get_eventid_for_filename(evt)
        #arr.append([a,locdata['e1'],locdata['e2']])
        #print(arr)
        #raw_input("hola")
        orig_img = detector.raw(evt).astype(np.float32)
        img = (orig_img - old_dark).astype(np.int16)
        #h5_img_large = locdata['img_large']
        #raw_input( "hi check how different")
        #assert_equal(h5_img_large, img, "failed to reproduce large img result: xml=%s" % xml)
        
        img = scipy.misc.imresize(img.astype(np.float), 50).astype(np.int16)
        #h5_img_small = locdata['img_small']
        #assert_max_diff_less_than(h5_img_small, img, max_diff_thresh=2, failmsg="failed to reduce xml=%s" % xml)
        #raw_input("did we make it past stage 2")

        #orig_fc1, orig_fc2 = locdata['codeword1'], locdata['codeword2']
        #img = vgg16img2codeword.preProcessImg(img)
        img_prep = prep_for_vgg(img, channel_mean=8.46261599735)
        layers = sslearn.vgg16.get_model_layers(sslearn.session,imgs=img_prep, layer_names=['fc2'])

        

        codeword2 = layers[0][0,:]
        
        #report_diff(orig_fc2, codeword2, "fc2 with old_vgg16=%s on new small image")
        img = img_prep
        #raw_input("did we make it past stage 3?")
        #img = scipy.misc.imresize(img.astype(np.float32), (224, 224))
        #img = img.astype(np.float32)
        #img -= 8.4626
        print(img.shape)
        #raw_input("check out what vgg16img2codword did")
        sslearn.label(img, get_eventid_for_filename(evt))
        with open('list_record', 'wb') as f:
            pickle.dump(arr, f)
    #ch_mean = np.mean(shots_for_mean)
    #ii = 0    
    #while ii < 60:
    #    tm = event_times[event_order[ii]]
    #    ii += 1
    #    evt = run.event(tm)
    #    orig_img = detector.raw(evt)
    #    if orig_img is None:
    #        continue
    #    prep_img = prepare_image(orig_img, dark)
    #    prep_img -= ch_mean
    #    if prep_img is None:
    #        continue
    #    sslearn.label(prep_img, get_eventid_for_filename(evt))

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

