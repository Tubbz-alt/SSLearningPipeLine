import os
import matplotlib.pyplot as plt

plt.ion()

# Class with the following attributes:
#	outputdir: directory in which the output will go (typically transferLearning)
# 	output_prefix: experiment prefix (i.e. xppl3816)
# 	total_to_label: total number of images you want to label? Not sure where this is used... 
class SSLearnPipeline(object):
	def __init__(self, outputdir, output_prefix, total_to_label=50):

    		self.outputdir = outputdir
    		assert os.path.exists(outputdir)

    		self.output_prefix = output_prefix

    		self.labeled_dir = os.path.join(outputdir, 'labeled')
    		if not os.path.exists(self.labeled_dir):
      			os.mkdir(self.labeled_dir)

    		self.total_to_label = total_to_label

  	def update_label_file(self, label_file, shot_num, classification):
    		if classification == -1:
			return
		
		if os.path.exists(label_file):
			with open(label_file, 'r') as f:
				[yes,no] = [int(float(count)) for count in f.readline().strip('\n').split(' ')]
    		else:
			yes = 0
			no = 0
    		if classification==0: 
			yes += 1
    		elif classification==1: 
			no += 1

    		with open(label_file, 'w') as f:
			f.write(str(yes) + ' ' + str(no))

  	def label(self, img, img1, img2, delay, delay1, delay2, step, step1, step2, run_num, shot_num):

    		output_label_fname = os.path.join(self.labeled_dir, self.output_prefix + '_r' + str(run_num) + '_s' + str(shot_num) + '.dat')

		# Removed this since now we are tracking how many people agree and disagree with each image
    		#if os.path.exists(output_label_fname):
      		#	print("This image has already been checked - skipping: exists: %s" % output_label_fname)
      		#	return -1
   
		plt.style.use("dark_background")

    		if not img2 is None:

        		fig = plt.figure(1,figsize=(10,8))

        		# Current shot of interest
        		axes1 = fig.add_subplot(3,3,(2,6))
        		axes1.imshow(img)
        		plt.title('Predicted delay: ' + str(delay) + ', step: ' + str(step))
        		plt.axis('off')
        		axes2 = fig.add_subplot(3,3,(1,4))
        		plt.title('Current shot', y=0.5)
        		plt.axis('off')

        		# Past 2 shots agreed with
        		axes3 = fig.add_subplot(3,3,8)
        		axes3.imshow(img1)
        		plt.title('delay: ' + str(delay1) + ', step: ' + str(step1))
        		plt.axis('off')
        		axes4 = fig.add_subplot(3,3,9)
        		axes4.imshow(img2)
        		plt.title('delay: ' + str(delay2) + ', step: ' + str(step2))
        		plt.axis('off')
        		axes5 = fig.add_subplot(3,3,7)
        		plt.title('Reference shots', y=0.5)
        		plt.axis('off')
        		plt.subplots_adjust(left=0.0)
        		plt.show()

    		else:
        		plt.figure(1)
        		plt.imshow(img)
        		plt.title('Predicted delay: ' + str(delay) + ', step: ' + str(step))
        		plt.show()

    		ans = raw_input("Hit enter to agree with this image, or n to reject it: ")
      		if ans.lower().strip()=='':
			classification = 0
      		elif ans.lower().strip()=='n':
        		classification = 1
		else: classification = -1
		plt.close(1)
    
		self.update_label_file(output_label_fname, shot_num, classification)
    		return classification
