from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.externals import joblib

from . import util
from . import vgg16

plt.ion()


class SSLearnPipeline(object):
  def __init__(self, outputdir, 
               output_prefix, 
               vgg16_weights,
               max_boxes_in_one_image, 
               tensorflow_session = None,
               total_to_label=50):
    self.outputdir = outputdir
    print(self.outputdir)
    assert os.path.exists(outputdir)
    self.output_prefix = output_prefix
    assert os.path.exists(vgg16_weights)
    self.vgg16_weights = vgg16_weights
    self.labeled_dir = os.path.join(outputdir, 'labeled')
    if not os.path.exists(self.labeled_dir):
      os.mkdir(self.labeled_dir)
    self.jpegs_to_label = os.path.join(outputdir, 'jpegs_to_label')
    if not os.path.exists(self.jpegs_to_label):
      os.mkdir(self.jpegs_to_label)
    self.total_to_label = total_to_label
    self.max_boxes_in_one_image = max_boxes_in_one_image
    if tensorflow_session is None:
      tensorflow_session = tf.Session()
    self.imgs = imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    self.session = tensorflow_session
    self.vgg16 = vgg16.vgg16(imgs=self.imgs,
                             weights=self.vgg16_weights,
                             sess=self.session,
                             trainable=False,
                             stop_at_fc2=True)

  def get_category(self, boxes_labeled):
    assert len(boxes_labeled) <= self.max_boxes_in_one_image
    assert len(boxes_labeled) >= 0
    if boxes_labeled == []: return 0
    if boxes_labeled == [1]: return 1
    if boxes_labeled == [2]: return 2
    if boxes_labeled == [3]: return 3
    raise Exception("not fully implemented")

  def labeling_not_done(self):
    number_labeled = len(glob.glob(os.path.join(self.labeled_dir, '%s*.json' % self.output_prefix)))
    if number_labeled >= self.total_to_label:
      return False
    return True

  def make_labelme_command_line(self, input_jpeg_fname, output_label_fname):
    import labelme
    import labelme.app as labelme_app
    cmd = 'python %s' % labelme_app.__file__
    cmd += ' --output %s' % output_label_fname
    cmd += ' %s' % input_jpeg_fname
    return cmd

  def validate_label_file(self, label_file):
    label_info = json.load(file(label_file,'r'))
    shapes = label_info['shapes']
    assert len(shapes)<=self.max_boxes_in_one_image, "there are more than %d boxes labeled" % self.max_boxes_in_one_image
    unique_labels = set()
    for shape in shapes:
      try:
        shape_label = int(shape['label'])
      except:
        raise Exception('all shape labels must be a 0-up integer, i.e, 0,1,2, etc, but this label is %s' % shape['label'])
      assert shape_label >= 0 and shape_label < self.max_boxes_in_one_image
      unique_labels.add(shape_label)
      assert util.is_closed_five_point_box(shape['points'])
    assert len(unique_labels)==len(shapes), "all box labels must be unique"

  def update_label_file(self, label_file, keystr, codeword):
    label_info = json.load(file(label_file,'r'))
    del label_info['imageData']
    category = 0
    for shape in label_info['shapes']:
      box_id = int(shape['label'])
      category += 1<<box_id
    label_info['category'] = str(category)
    label_info['imgkey'] = keystr
    strfile = self.outputdir + '/codeword/'+keystr
    
    np.save(strfile,codeword)
    #label_info['codeword'] = np.array_str(codeword)
    #added this for codeword
    # TODO: should be embed the codeword in the json? Or start creating an hdf5 file somewhere
    # and reference where the codeword is?

    fout = file(label_file,'w')
    fout.write(json.dumps(label_info, sort_keys=True, indent=4, separators=(',' , ':' )))
    fout.close()
               
  def label(self, img, keystr):
    plt.figure(1)
    plt.imshow(img[0,:,:,1])
    plt.show()
    plt.pause(.1)

    output_label_fname = os.path.join(self.labeled_dir, self.output_prefix + '_' + keystr + '.json')
    output_jpeg_fname  = os.path.join(self.jpegs_to_label, self.output_prefix + '_' + keystr + '.jpeg')

    if os.path.exists(output_label_fname):
      print("This image has already been labeled - skipping: exists: %s" % output_label_fname)
      return None
    
    while True:
      #   then user can skip classes already labeled
      ans = raw_input("Hit enter to label this image, or n to skip it: ")
      if ans.lower().strip()=='':
        break
      if ans.lower().strip()=='n':
        return None
        
    a = np.max(img)
    b = np.min(img)
    print(a)
    print("hiiii")
    print(b)
    print(img.shape)
    img_prep= img;
    util.create_jpeg(img_prep, output_jpeg_fname)
    labelme_command_line = self.make_labelme_command_line(input_jpeg_fname=output_jpeg_fname,
                                                          output_label_fname=output_label_fname)
    print("about to execute:\n  %s" % labelme_command_line)
    assert 0 == os.system(labelme_command_line)
    assert os.path.exists(output_label_fname)
    
    self.validate_label_file(output_label_fname)
    #img_batch_for_vgg16, orig_resize_mean = util.prep_img_for_vgg16(img, mean_to_subtract=None)
    # TODO: get a more accurate mean to subtract? Keep track of the orig_resize_mean?

    #layers = self.vgg16.get_model_layers(self.session,imgs=[img_batch_for_vgg16], layer_names=['fc2'])

    layers = self.vgg16.get_model_layers(self.session, imgs = img, layer_names = ['fc2'])

    codeword = layers[0][0,:]
    assert codeword.shape == (4096,)
    
    self.update_label_file(output_label_fname, keystr, codeword=codeword)

    # TODO: print category given for this image
    #       print balance - i.e, how many of each category are currently labeled
    #       then user can skip over-sampled categories

  def build_models(self):
    pass
    #print( "holaaa")
    json_string_prep = self.outputdir + '/labeled/*.json'
    #print(json_string_prep)
    
    json_files = glob.glob(json_string_prep)
    #print(json_files)
    
    size = len(json_files)
    features_class = np.zeros((size,5005))
    i =0 
    ci = 0
    for k in json_files:
      label_file = k
      label_info = json.load(file(label_file,'r'))
      #print(label_info['shapes'])
      #print(len(label_info['shapes']))
      l1 = 0 ; l2 = 0; l3 = 0; l4 = 0; l5 = 0; l6 = 0; l7 =0; l8 = 0;
      if len(label_info['shapes'])==2:
           ci = 3
           for k in range(0,1):
             l = int(label_info['shapes'][k]['label'])
             if(l == 0):
               l1 = label_info['shapes'][k]['points'][0][0]
               l2 = label_info['shapes'][k]['points'][0][1]
               l3 = label_info['shapes'][k]['points'][2][0]
               l4 = label_info['shapes'][k]['points'][2][1]
             if(l == 1):
               l5 = label_info['shapes'][k]['points'][0][0]
               l6 = label_info['shapes'][k]['points'][0][1]
               l7 = label_info['shapes'][k]['points'][2][0]
               l8 = label_info['shapes'][k]['points'][2][1]
      if len(label_info['shapes'])==0:
           ci = 0
      if len(label_info['shapes']) ==1:
           l = int(label_info['shapes'][0]['label'])
           ci = l +1
           if(l == 0):
             l1 = label_info['shapes'][0]['points'][0][0]
             l2 = label_info['shapes'][0]['points'][0][1]
             l3 = label_info['shapes'][0]['points'][2][0]
             l4 = label_info['shapes'][0]['points'][2][1]
           if(l == 1):
             l5 = label_info['shapes'][0]['points'][0][0]
             l6 = label_info['shapes'][0]['points'][0][1]
             l7 = label_info['shapes'][0]['points'][2][0]
             l8 = label_info['shapes'][0]['points'][2][1]
           
      strfile = self.outputdir + '/codeword/'+label_info['imgkey']+'.npy'
      code = np.load(strfile)
      features_class[i,0:4096] = code
      features_class[i,4096] = ci
      features_class[i,4097] = l1
      features_class[i,4098] = l2
      features_class[i,4099] = l3
      features_class[i,4100] = l4
      features_class[i,4101] = l5
      features_class[i,4102] = l6
      features_class[i,4103] = l7
      features_class[i,4104] = l8
      i = i +1

    #Classifier Model
    print('ClassifierModel')
    trial = np.zeros((5))
    for num in range(0,5):
      reg = 10**(num/2)
      reg = 1/reg
      model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=reg, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=100, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=1)
      unique,counts = np.unique(features_class[:,4096], return_counts = True)
      mincounts = np.min(counts)
      print(counts)
      #assert mincounts>1, "Only 1 example of a certain label. Need atleast 2 examples"
        
      #model.fit(features_class[:,0:4096],features_class[:,4096])
      #joblib.dump(model,'ClassifierModel.pkl')
      #Measure Model Performance
      total = 0
      #indices = np.arange(features_class.shape[0])
      #for k in range(0,features_class.shape[0]):
      #  data = features_class[indices!=k,:]
      #  model.fit(data[:,0:4096],data[:,4096])
      #  w = model.predict(features_class[k,0:4096])
      #  total = total + (w==features_class[k,4096])
      #accuracy = total/features_class.shape[0]
      #trial[num] = accuracy
      #print("accuracy is ")
      #print(accuracy)
    #r = np.argmax(trial)
    #reg = 10**(r/2)
    #reg = 1/reg
    #model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=reg, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=100, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=1)
    #model.fit(features_class[:,0:4096],features_class[:,4096])
    #joblib.dump(model,'ClassifierModel.pkl')

    #Class One Regression 
    #print('Class One Regressor Model')
    #raw_input("ok check")
    data = np.where(features_class[:,4096]==1)
    data = features_class[data[0],:]
    #print(data.shape)
    model = MultiOutputRegressor(LinearRegression())
    train = 60
    model = model.fit(data[0:60,0:4096],data[0:60,4097:4101])
    r = model.predict(data[60:,0:4096])
    temp = np.sum(np.abs((r - data[60:,4097:4101])))
    #print(np.abs(r - data[60:,4097:4101]))
    #print(temp)
    t = np.abs(r - data[60:,4097:4101])
    #ipython
    #print(1/0)
    #print(r.shape)
    t = t.reshape(t.shape[0] * t.shape[1])
    #print(t.shape)
    #print(data[0:60,4097:4101].shape)
    hist, bin_edges = np.histogram(t,density = False)
    #print(t)
    #print(hist)
    #print(bin_edges)
    plt.hist(t,bins = 'auto')
    plt.title("x-axis:error , y-axis:number of entries")
    raw_input("lets view the results")
    joblib.dump(model,'ClassOneRegression.pkl')
    #total = 0
    #indices = np.arange(features_class.shape[0])
    #for k in range(0,features_class.shape[0]):
    #  data = features_class[indices!=k,:]
    #  model.fit(data[:,0:4096],data[:,4097:4101])
    #  w = model.predict(features_class[k,0:4096])
    #  total = total + np.sum((w-features_class[k,4097:4101])**2)
    #accuracy1 = total/features_class.shape[0]
    #Class Two Regression 
    data = np.where(features_class[:,4096]==2)
    data = features_class[data[0],:]
    model = MultiOutputRegressor(LinearRegression())
    model = model.fit(data[:,0:4096],data[:,4101:4105])
    joblib.dump(model,'ClassTwoRegression.pkl')
    total = 0
    indices = np.arange(features_class.shape[0])
    #for k in range(0,features_class.shape[0]):
    #  data = features_class[indices!=k,:]
    #  model.fit(data[:,0:4096],data[:,4101:4105])
    #  w = model.predict(features_class[k,0:4096])
    #  total = total + np.sum((w-features_class[k,4101:4105])**2)
    #accuracy2 = total/features_class.shape[0]
    #Class Three Regression
    data = np.where(features_class[:,4096]==3)
    data = features_class[data[0],:]
    model = MultiOutputRegressor(LinearRegression())
    model = model.fit(data[:,0:4096],data[:,4097:4105])
    joblib.dump(model,'ClassThreeRegression.pkl')
    #total = 0
    #indices = np.arange(features_class.shape[0])
    #for k in range(0,features_class.shape[0]):
    #  data = features_class[indices!=k,:]
    #  model.fit(data[:,0:4096],data[:,4097:4105])
    #  w = model.predict(features_class[k,0:4096])
    #  total = total + np.sum((w-features_class[k,4097:4105])**2)
    #accuracy3 = total/features_class.shape[0]


     
      
  def predict(self, img):
    img_batch_for_vgg16, jnk = util.prep_img_for_vgg16(img, mean_to_subtract=None)
    #model = clf = joblib.load('ClassifierModel.pkl') 
    layers = self.vgg16.get_model_layers(self.session, 
                                         imgs=img_batch_for_vgg16, 
                                         layer_names=['fc2'])
    #Class Labeler
    model = joblib.load('ClassifierModel.pkl') 
    codeword = layers[0][0,:]
    category = model.predict(codeword)
    if(category==1):
      model = joblib.load('ClassOneRegression.pkl') 
      a = model.predict(codeword)
      pass
    if(category==2):
      model = joblib.load('ClassTwoRegression.pkl')
      a = model.predict(codeword)
      #print(a)
      pass
    if(category==3):
      model = joblib.load('ClassThreeRegression.pkl')
      a = model.predict(codeword)
      #print(a)
      pass
    prediction = {}
    prediction['failed'] = False
    prediction['category']=category
    if(category==1):
      prediction['boxes'][0]['x1'] = a[0]
      prediction['boxes'][0]['y1'] = a[1]
      prediction['boxes'][0]['x2'] = a[2]
      prediction['boxes'][0]['y2'] = a[3]
      
      pass
    if(category==2):
      prediction['boxes'][1]['x1'] = a[0]
      prediction['boxes'][1]['y1'] = a[1]
      prediction['boxes'][1]['x2'] = a[2]
      prediction['boxes'][1]['y2'] = a[3]
      #print(a)
      pass
    if(category==3):
      prediction['boxes'][0]['x1'] = a[0]
      prediction['boxes'][0]['y1'] = a[1]
      prediction['boxes'][0]['x2'] = a[2]
      prediction['boxes'][0]['y2'] = a[3]
      prediction['boxes'][1]['x1'] = a[4]
      prediction['boxes'][1]['y1'] = a[5]
      prediction['boxes'][1]['x2'] = a[6]
      prediction['boxes'][1]['y2'] = a[7]
    #prediction['category_confidence'] = 0.99
    #prediction['boxes'] = {0:None, 1:None} # user may look for None for any box
    #prediction['boxes'][0] = {'confidence':0.99,
    #                          'x1':1,
    #                          'x2':10,
    #                          'y1':2,
    #                          'y2':10}
    #prediction['boxes'][1] = {'confidence':0.99,
    #                          'x1':101,
    #                          'x2':110,
    #                          'y1':102,
    #                          'y2':110}
    return prediction

  
  

  
  
