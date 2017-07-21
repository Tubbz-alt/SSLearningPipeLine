import os
import matplotlib.pyplot as plt

plt.ion()

# Class with the following attributes:
#	outputdir: directory in which the output will go (typically transferLearning)
# 	output_prefix: experiment prefix (i.e. xppl3816)
# 	total_to_label: total number of images you want to label? Not sure where this is used... 
class SSLearnPipeline(object):
  def __init__(self, outputdir, 
               output_prefix, 
               total_to_label=50):

    self.outputdir = outputdir
    assert os.path.exists(outputdir)

    self.output_prefix = output_prefix

    self.labeled_dir = os.path.join(outputdir, 'labeled')
    if not os.path.exists(self.labeled_dir):
      os.mkdir(self.labeled_dir)

    self.pngs_to_label = os.path.join(outputdir, 'pngs_to_label')
    if not os.path.exists(self.pngs_to_label):
      os.mkdir(self.pngs_to_label)

    self.total_to_label = total_to_label

  def update_label_file(self, label_file, shot_num):
    fout = file(label_file,'w')
    fout.write(str(shot_num) + '\n')
    fout.close()
               
  def label(self, img, run_num, shot_num):
    plt.figure(1)
    plt.imshow(img[:,:,:])
    plt.show()
    plt.pause(.1)

    output_label_fname = os.path.join(self.labeled_dir, self.output_prefix + '_r' + str(run_num) + '_s' + str(shot_num) + '.dat')

    if os.path.exists(output_label_fname):
      print("This image has already been checked - skipping: exists: %s" % output_label_fname)
      return None
    
    while True:
      #   then user can skip classes already labeled
      ans = raw_input("Hit enter to agree with this image, or n to reject it: ")
      if ans.lower().strip()=='':
        break
      if ans.lower().strip()=='n':
        return None
        
    self.update_label_file(output_label_fname, shot_num)

