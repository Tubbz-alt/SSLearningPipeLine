import numpy as np  
import scipy.misc
from sslearnpipeline import SSLearnPipeline
import os
import argparse

def main(run_num):
    
    outputdir = '/reg/neh/home/kfotion/work/transferLearning'
	
    sslearn = SSLearnPipeline(outputdir=outputdir,
                              output_prefix='xppl3816')

    A = np.load('indexlist.npy')

    for index in A:
	
	filename = os.path.join(outputdir, 'pngs_to_label/r' + str(run_num) + '_s' + str(index)  + '_vi.png')
	orig_img = scipy.misc.imread(filename)
        sslearn.label(orig_img, run_num, index)

if __name__ == '__main__':

    helpstr = 'Display time tool results and receive input'
    parser = argparse.ArgumentParser(description=helpstr);
    parser.add_argument('-r','--run',dest='run_num',type=int, help='run number', default=110)

    args = parser.parse_args();

    run_num = args.run_num
 
    main(run_num)

