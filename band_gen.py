from collections import defaultdict
from keras.datasets import mnist
import numpy as np
import pickle
import random
import time
import os

def band_gen(im, start_pos, end_pos, new_val, same):
	if same == True:
		for line in range(0, im.shape[1]):
			im[line][start_pos:end_pos] = random.uniform(new_val[0],new_val[1])

	else:
		for line in range(0, im.shape[1]):
			for col in range(start_pos, end_pos+1):
				im[line][col] = random.uniform(new_val[0],new_val[1])

	return im


def band_gen_run(start_pos, end_pos, new_val, same, tot_im, dump, dataset=mnist.load_data()):
	(x, y), (x_test, y_test) = dataset

	start = time.time()
	print("\nModifying train images...")
	mod_x = [band_gen(x_i, start_pos, end_pos, new_val, same) for x_i in x[0:tot_im]]
	mod_y = y[0:tot_im]
	print("Done\n")
	print("Modifying test images...")
	mod_x_test = [band_gen(x_i, start_pos, end_pos, new_val, same) for x_i in x[0:tot_im]]
	mod_y_test = y_test[0:tot_im]
	print("Done\n")

	if dump == True:
		name = str(start_pos) + "_" + str(end_pos) + "_" + str(new_val) + "_" + str(same) + "_" + str(tot_im)
		if not os.path.exists("./data/"):
			os.makedirs("data")
		print("Dumping into file...")
		pickle.dump( ((mod_x, mod_y), (mod_x_test, mod_y_test)), open( "./data/mnist_band_gen_" + name + ".p", "wb" ))
		print("Done\n")
	
	end = time.time()
	print("Run time: " + str(round(end - start, 3)) + "s")

	return ((mod_x, mod_y),(mod_x_test, mod_y_test))