# Trent VanSlyke, last updated 5/12/2021
# Written as the final project for Algorithms at SUNY Polytechnic Institute, Spring 2021
# Project goal: Write a generative adversarial network used to process video into higher framerates

# Based on the work of: 
# Yoon, Nam, et al. (25 March 2021). Frame-rate Up-conversion Detection Based on Convolutional Neural Network for Learning Spatiotemporal Features. 
# Retreived from https://arxiv.org/pdf/2103.13674.pdf, May 5 2021.

import cv2
import sys
import math
import os.path
import os
import numpy as np
import tensorflow as tf

# Steps

# Preprocessing
# Takes input videos, performs a bilinear interpolation on the source video, and then processes the individual
# frames into something the neural network is expecting.
# According to the paper (Noon et al):
# * Step 1 is extracting six consecutive frames selected from a random starting position in the video. The feature map is (frames x channels x height x width),
#   or t x c x h x w, or 6x3x256x256. In my network, I use 6x3x256x256
# * Step 2 is converting the RGB frames in each 3 channels into a gray imnage with one channel. Luminance = (R * 0.299 + G * 0.587 + B * 0.114). This, in effect,
#   reduces dimensionality to 6x256x256
# * Step 3 is creating the residual frame r, as the difference between consequetive luminance frames from step 2. This turns 6x256x256 to 5x256x256, as it is the
#   change between frames. 

# I is defined as input frames comprising six sequential frames extracted from the video.
# Hrfg is defined as the residual frame generation module (see: preprocessing)
# Ir = Hrfg(I) and is 5 residual input frames 
# Fru = Hrfe(Ir) (Hrfe = residual feature extraction module)
# Fsu = Hsfe(Fru) (Hsfe = spatiotemporal feature extraction module)

# In essence: 
# video
# -> I (6 original frames, 6x3x256x256) -> preprocess (residual frame generation)
# -> Ir (5 residual information frames, 5x256x256) -> residual feature extraction module (block 1)
# -> Fru (Frames-residual features) -> spatiotemporal feature extraction module (block 2)
# -> classification -> Original or forged

# Input: data/ folder contains videos labeled "n.mp4" for the original video and "n-dain.mp4" for DAIN upscaled
# video.
def get_frames_from_video(video_name):
	print("Loading " + video_name + "...")
	if not os.path.exists("data\\" + video_name):
		print("File doesn't exist! Exiting early!")
		quit()
	video = cv2.VideoCapture("data\\" + video_name)
	print("Loaded video. Extracting frames...")
	frames = []
	while True:
		(is_finished, frame) = video.read()
		if not is_finished:
			break
		frames.append(frame)
	video.release()
	print("Extracted frames. Count: " + str(len(frames)))
	return frames

def bilinear_between_frames(frame1, frame2):
	height, width, channels = frame1.shape
	intermediate = np.zeros((height, width, channels), np.uint8)
	for x in range(0, len(intermediate)):
		intermediate[x] = (frame1[x] + frame2[x]) / 2
	return intermediate

# This will return len(frames) * 2 - 1
def bilinear_upscale(frames):
	frame_count = (len(frames) * 2)
	frame_count -= 2 + (frame_count % 2)
	new_frames = [None] * frame_count
	for x in range(0, frame_count):
		preframe = math.floor(x/2)
		if x%2 == 0:
			new_frames[x] = bilinear_between_frames(frames[preframe], frames[preframe+1])
		else:
			new_frames[x] = frames[preframe]
	return new_frames

def greyscale(data):
	new_frames = [None] * len(data)
	for x in range(0, len(data)):
		frame = data[x]
		height, width, channels = frame.shape
		intermediate = np.zeros((height, width), np.uint8)
		for x in range(0, len(intermediate)):
			location = x * 3
			intermediate[x] = (data[location]* 0.299) + (data[location+1] + 0.587) + (data[location+2] * 0.114)
		new_frames[x] = intermediate
	return new_frames

def difference(data):
	new_frames = [None] * (len(data)-1)
	for x in range(0, len(new_frames)):
		frame = data[x]
		height, width, channels = frame.shape
		new_frames[x] = np.zeroes((height, width, 1), np.uint8)
		for y in range(0, len(new_frames[x]))
			new_frames[x][y] = math.abs(data[x][y] - data[x+1][y])
	return new_frames

def main(number_of_sources):
	data_original = []
	data_bilinear = []
	data_dain = []
	height = 0
	width = 0
	channels = 0
	# Preprocessing
	# Loads data from sources on disk, processes them, and stores in the data_* variables
	for source in range(1, number_of_sources + 1):
		#dain = get_frames_from_video(str(source) + ".mp4")
		original = get_frames_from_video(str(source) + ".mp4")

		height, width, channels = original[0].shape
		
		print("Creating downscaled video (60fps -> 15fps)...")
		low_fps_count = math.floor(len(original) / 4)
		low_fps_frames = [None] * low_fps_count
		for i in range(0, low_fps_count):
			low_fps_frames[i] = original[i * 4]
		print("Downscaled video created. Frame count: " + str(len(low_fps_frames)))

		print("Creating bilinear upscaled video (15fps -> 30fps)...")
		bilinear_temp_frames = bilinear_upscale(low_fps_frames)
		print("Upscaled video created. Frame count: " + str(len(bilinear_temp_frames)))
		print("Creating bilinear upscaled video (30fps -> 60fps)...")
		bilinear_frames = bilinear_upscale(bilinear_temp_frames)
		print("Upscaled video created. Frame count: " + str(len(bilinear_frames)))
		
		new_frame_count = min(len(original), len(bilinear_frames), len(dain))

		print("Bilinear interpolation: lost " + str(len(original) - len(bilinear_frames)) + " frames.")
		print("New frame count: " + str(new_frame_count))

		data_original.append(original[0:new_frame_count])
		data_bilinear.append(bilinear_frames[0:new_frame_count])
		data_dain.append(dain[0:new_frame_count])

	# Residual Feature Learning (block 1)
	# Build dataset from known frames
	print("Creating dataset...")
	delta_original = difference(greyscale(data_original))
	delta_dain = difference(greyscale(data_dain))
	delta_bilinear = difference(greyscale(data_bilinear))

	number_of_sources = 10
	train_images = []
	train_images.append(delta_original)
	train_images.append(delta_dain)
	train_images.append(delta_bilinear)
	train_labels = []
	train_labels.append([1] * len(delta_original))
	train_labels.append([0] * len(delta_dain))
	train_labels.append([0] * len(delta_bilinear))
	test_images = []
	test_labels = []

	print("Dataset has been created. Building model for block 1...")

	# Residual features

	input_residual = tf.keras.layers.Input(shape=(5, height, width))
	block1_first_convolution = tf.keras.layers.Conv2D(filters=60, kernel_size=3, groups=5, padding="same", input_shape=(5,height,width)) # Group convolution
	block1_convolutions = tf.keras.models.Sequential([
		tf.keras.layers.BatchNormalization(), # Aims for output=0, std deviation = 1
		tf.keras.layers.Conv2D(filters=60, kernel_size=3, groups=5, activation="relu"), # Group convolution
		tf.keras.layers.BatchNormalization(), # Aims for output=0, std deviation = 1
		tf.keras.layers.Conv2D(filters=60, kernel_size=3, groups=5, activation="relu"), # Group convolution
	])
	block1 = tf.keras.models.Sequential([
		block1_first_convolution,
		block1_convolutions,
		tf.keras.layers.Add()((block1_first_convolution)(input_residual))
	])
	residual_feature_learning_module = tf.keras.models.Sequential([
		block1, block1, block1, block1, block1
	])

	print("Building model for block 2...")
	# Spatiotemporal
	input_spatio = tf.keras.layers.Input(shape=(5, height, width))
	block2_first_convolution = tf.keras.layers.SeparableConv2D(filters=60, kernel_size=3, groups=5, padding="same", input_shape=(5,height,width))
	block2_convolutions = tf.keras.models.Sequential([
		tf.keras.layers.BatchNormalization(), # Aims for output=0, std deviation = 1
		tf.keras.layers.DepthwiseConv2D(filters=60, kernel_size=3, groups=5, activation="relu"), # Group convolution
		tf.keras.layers.BatchNormalization(), # Aims for output=0, std deviation = 1
		tf.keras.layers.DepthwiseConv2D(filters=60, kernel_size=3, groups=5, activation="relu"), # Group convolution
	])
	block2 = tf.keras.models.Sequential([
		block2_first_convolution,
		block2_convolutions,
		tf.keras.layers.Add()((block2_first_convolution)(input_spatio))
	])
	spatiotemporal_feature_learning_module = tf.keras.models.Sequential([
		block2, block2, block2, block2, block2
	])

	# Bring it all together
	print("Building complete model...")
	complete_network = tf.keras.models.Sequential([
		residual_feature_learning_module,
		spatiotemporal_feature_learning_module
	])

	print("Complete model built. Compiling and fitting...")
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	complete_network.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
	complete_network.fit(train_images, train_labels, epochs=50)
	complete_network.evaluate(train_images, train_labels, verbose=2)
main(1)
