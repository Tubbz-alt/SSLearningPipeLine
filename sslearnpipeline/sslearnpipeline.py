import os
import matplotlib.pyplot as plt
import matplotlib.lines as mplines
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

	def update_label_file(self, label_file, classification):
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

  	def label(self, predictions, img_of_interest, reference_imgs, run_num):

		(img, index) = img_of_interest
    		output_label_fname = os.path.join(self.labeled_dir, self.output_prefix + '_r' + str(run_num) + '_s' + str(index) + '.dat')

		plt.style.use("dark_background")

		(pixel_pos,delay,step) = predictions.get(index)
    		ax = plot_predictions(predictions, step)

		if len(reference_imgs) > 1:
			(img1,index1) = reference_imgs[0]
			(img2,index2) = reference_imgs[1]
			(pixel_pos1,delay1,step1) = predictions.get(index1)
                	(pixel_pos2,delay2,step2) = predictions.get(index2)
			fig = plot_multiple(ax, (img,index,pixel_pos,delay,step), (img1,index1,pixel_pos1,delay1,step1), (img2,index2,pixel_pos2,delay2,step2))
    		else:
			fig = plot_single(ax, img, index, pixel_pos, delay, step)

    		ans = raw_input("Hit enter to agree with this image, or n to reject it: ")
      		if ans.lower().strip()=='':
			classification = 0
      		elif ans.lower().strip()=='n':
        		classification = 1
		else: classification = -1
		fig.clear()
		ax.clear()
    
		self.update_label_file(output_label_fname, classification)

    		return classification


def plot_multiple(ax, (img,index,pixel_pos,delay,step), (img1,index1,pixel_pos1,delay1,step1), (img2,index2,pixel_pos2,delay2,step2)):

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

	# Plot where this shot lies amongst others
	if not ax is None:
		ax.scatter(index,pixel_pos,delay, c='r', s=200)
		#ax.scatter(index1,pixel_pos1,delay1, c='g',s=200)
		#ax.scatter(index2,pixel_pos2,delay2, c='g',s=200)	

	return fig	

def plot_single(ax, img, index, pixel_pos, delay, step):

	fig = plt.figure(1)
	plt.imshow(img)
	plt.title('Predicted delay: ' + str(delay) + ', step: ' + str(step))
	plt.show()

	# Plot where this shot lies amongst others
	if not ax is None:
		ax.scatter(index, pixel_pos, delay, c='r', s=200)

	return fig

def plot_predictions(predictions, step_of_interest):

	sub_pred = {key:value for key,value in predictions.iteritems() if value[2] == step_of_interest}

	fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        xs = sub_pred.keys()
        ys = zip(*sub_pred.values())[0]
        zs = zip(*sub_pred.values())[1]
	ax.scatter(xs, ys, zs, c='b')
	plt.title('Predictions of all shots from step ' + str(step_of_interest))
	ax.set_xlabel('shot_num')
	ax.set_ylabel('pixel_pos')
	ax.set_zlabel('pred_delay')

	# Enable legend
	scatter1_proxy = mplines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
	scatter2_proxy = mplines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
	ax.legend([scatter1_proxy, scatter2_proxy], ['shot of interest', 'all other shots'], numpoints = 1, loc='lower left', ncol=1)
	#scatter3_proxy = mplines.Line2D([0],[0], linestyle='none', c='g', marker = 'o')
	#ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy], ['shot of interest', 'all other shots', 'reference shots'], numpoints = 1, loc='lower left', ncol=1)

	# Set perspective
	ax.view_init(10, 10)

	return ax
