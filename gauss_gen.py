from scipy.ndimage.filters import gaussian_filter
from keras.datasets import mnist
import pickle
import time

def gauss_gen(sig, tot_im, dump, dataset=mnist.load_data()):
	(x, y), (x_test, y_test) = dataset

	start = time.time()
	print("\nBlurring train images...")
	blurred_x = [gaussian_filter(x_i , sigma=sig) for x_i in x[0:tot_im]]
	blurred_y = y[0:tot_im]
	print("Done\n")
	print("Blurring test images...")
	blurred_x_test = [gaussian_filter(x_i , sigma=sig) for x_i in x_test[0:tot_im]]
	blurred_y_test = y_test[0:tot_im]
	print("Done\n")

	if dump == True:
		print("Dumping into file...")
		pickle.dump( ((blurred_x, blurred_y), (blurred_x_test, blurred_y_test)), open( "./data/mnist_blurred_" + str(sig) + "_" + str(tot_im) + ".p", "wb" ))
		print("Done\n")

	end = time.time()
	print("Run time: " + str(round(end - start, 3)) + "s")

	return ((blurred_x, blurred_y),(blurred_x_test, blurred_y_test))