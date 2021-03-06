"""
SSLearningPipeline Main Script
"""
############
# Standard #
############
import os
import logging
from logging.handlers import RotatingFileHandler
import argparse
from pathlib import Path
import re
import random

###############
# Third Party #
###############
import numpy as np  
from scipy.misc import imread
import cv2 

##########
# Module #
##########
from sslearnpipeline import SSLearnPipeline

def access_predictions(run_num, model):
	with open('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/xppl3816_r' + str(run_num) + '_' + model + '_plot.dat', 'r') as f:
        rf_lines = f.readlines()
	with open('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/xppl3816_r' + str(run_num) + '_delays.dat','r') as f:
		step_lines = f.readlines()
	
	# Build prediction dictionary: {shot#: (pixel_pos, delay, step)}
        predictions = {int(float(line.split(' ')[0])): (float(line.split(' ')[2]), float("%.4f" % float(line.strip('\n').split(' ')[3])), int(float(step_lines[int(float(line.split(' ')[0]))].split(' ')[0]))) for line in rf_lines}

       	return predictions

def main(args):
    """
    Main SSleariningPipeline labeling script.
    """
    # Instantiate the labeler
    sslearn = SSLearnPipeline(outputdir=str(args.output), 
                              output_prefix=args.experiment)

    # Grab indices
    all_image_files = get_images_from_dir(os.path.join(str(args.images), str(args.run)), shuffle=True)
    
    # Access all delays and pixel positions and plot
    predictions = access_predictions(args.run, args.model)

    # Meta vars
    exit = False                          # Check if we should exit

    # Begin looping through images
    reference_imgs = []
    for filename in all_image_files:
        while not exit:
            try:
                
		filepath = filename
		index = int(float(re.sub('r.*_s','',filepath.parts[-1].rstrip('_vi.png'))))
		
		# Make sure it exists
                if not filepath.exists():
                    logger.warn("File '{0}' does not exist. Skipping.".format(
                        str(filepath)))
                    break
                # Read and label the image
		img_of_interest = (imread(str(filepath)), index)
		if index in predictions:
			classification = sslearn.label(predictions, img_of_interest, reference_imgs, args.run)
                	if classification==0:
				reference_imgs = [img_of_interest if i == 0 else reference_imgs[i-1] for i in range(0, min(2, len(reference_imgs)+1))]
		else:
			logger.warn("File '{0}' has too low of intensity. Skipping".format(
			   str(filepath)))
		break

            # Handle Interrupts
            except (KeyboardInterrupt, EOFError):
                try:
                    print                 # Add a new line
                    logger.info("Labeler interrupted.")
                    while not exit:
                        inp = raw_input("Enter 'c' to continue, 'e' to exit: ")
                        if inp.lower() == "c":
                            break
                        elif inp.lower() == "e":
                            exit = True
                        else:
                            logger.info("Invalid input '{0}'".format(inp))

                # Exit if another keyboard interrupt is encountered
                except (KeyboardInterrupt, EOFError):
                    exit = True
                finally:
                    print                 # Add a new line

        # End before completing the the loop
        if exit:
            break

def get_images_from_dir(target_dir, n_images=None, shuffle=False, out_type=list,
                        recursive=False, read_mode=cv2.IMREAD_GRAYSCALE,
                        glob="*"):
    """
    Crawls through the contents of inputted directory and returns a list of
    paths to the images.
    """
    image_ext = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])
    target_dir_path = Path(target_dir)
    if recursive and glob == "*":
         glob = "**"
    # Grab path of all image files in dir
    image_paths = [p for p in sorted(target_dir_path.glob(glob)) if
                    p.is_file() and p.suffix[1:].lower() in image_ext]
    
    # Shuffle the list of paths
    if shuffle:
	random.shuffle(image_paths)
    	if out_type is dict:
    		logger.warning("Shuffle set to True for requested output type dict")
    # Only keep n_images of those files
    if n_images:
    	image_paths = image_paths[:n_images]

    return image_paths
        
def get_logger(name, stream_level=logging.warn, log_file=True, 
               log_dir=Path("."), max_bytes=1024*1024):
    """
    Returns a properly configured logger that has a stream handler and a file
    handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # One format to display the user and another for debugging
    format_stream = "%(levelname)-2s: %(message)4s"
    format_debug = "%(asctime)s:%(filename)s:%(lineno)4s - " \
      "%(funcName)s():    %(levelname)-8s %(message)4s"
    # Prevent logging from propagating to the root logger
    logger.propagate = 0

    # Setup the stream logger
    console = logging.StreamHandler()
    console.setLevel(stream_level)
    # Print log messages nicely if we arent in debug mode
    if stream_level >= logging.INFO:
        stream_formatter = logging.Formatter(format_stream)
    else:
        stream_formatter = logging.Formatter(format_debug)
    console.setFormatter(stream_formatter)
    logger.addHandler(console)
    
    if log_file:
        log_file = log_dir / "log.txt"
        # Create the file if it doesnt already exist
        if not log_file.exists():
            log_file.touch()
        # Setup the file handler
        file_handler = RotatingFileHandler(
            str(log_file), mode='a', maxBytes=max_bytes, backupCount=2,
            encoding=None, delay=0)
        file_formatter = logging.Formatter(format_debug)
        file_handler.setFormatter(file_formatter)
        # Always save everything
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger
                    
def check_args(args):
    """
    Checks that the inputted args are valid.
    """
    # # Check the various paths exist
    # Output dir
    if not args.output.exists():
        logger.warn("Output directory '{0}' does not exist. Creating new dir."
                    "".format(args.output))
        args.output.mkdir(parents=True)
    # Image dir
    if not args.images.exists():
        err_str = "Image directory '{0}' does not exist".format(args.images)
        logger.error(err_str)
        raise FileNotFoundError(err_str)
    # Log dir
    if not args.logdir.exists():
        args.logdir.mkdir(parents=True)
    return args

def setup_parser_and_logging(description=""):
    """
    Sets up the parser, by adding the arguments, parsing the inputs and then
    returning the args and parser.
    """
    # Setup the parser
    if not description:
        description = "Display time tool results and receive input."
    parser = argparse.ArgumentParser(description=description)

    # Add all the arguments
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase verbosity.",)
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run in debug mode.",)        
    parser.add_argument("-e", "--experiment", metavar="E", type=str,  
                        action="store", default="xppl3816", 
                        help="Experiment to run the labeler with.")
    parser.add_argument("-r", "--run", metavar="N", type=int,  default=110, 
                        action="store", help="Run number of the experiment.")
    parser.add_argument("-m", "--model", type=str,  default='RF',
                        action="store", help="Regression model to be used.")
    parser.add_argument("--logdir", metavar="P", type=str,  
                        default=str(Path.cwd()) + "/logs", action="store",
                        help="Path to save the logs in.")
    parser.add_argument("-o", "--output", metavar="P", type=str,  
                        default="/reg/d/psdm/XPP/xppl3816/scratch/transferLearning", action="store",
                        help="Path to save the labeled images in.")
    parser.add_argument("-i", "--images", metavar="P", type=str,
                        default="/reg/d/psdm/XPP/xppl3816/scratch/transferLearning" \
                        "/pngs_to_label/", action="store",
                        help="Path to get the images from.")
    # Parse the inputted arguments
    args = parser.parse_args()

    # Convert path strs to Paths
    args.output = Path(args.output)
    args.images = Path(args.images)
    args.logdir = Path(args.logdir)
    # Perform any argument checks
    args = check_args(args)

    # Set the amount of logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    # Get the logger
    logger = get_logger(__name__, stream_level=log_level, 
                        log_dir=args.logdir)
    logger.debug("Logging level set to {0}.".format(log_level))

    return args, logger

if __name__ == "__main__":
    # Parse arguments
    args, logger = setup_parser_and_logging()
    # Run the script
    main(args)

