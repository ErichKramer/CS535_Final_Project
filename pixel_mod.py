from collections import defaultdict
from keras.datasets import mnist
import numpy as np
import pickle
import random
import time


def pixel_mod(im, thr_cut, thr_new, nb, ev_oth):
	loc = np.where(im > thr_cut)
	dic = defaultdict(list)
	[dic[c].append(d) for c,d in zip(loc[0], loc[1])]
	dic_keys = dic.keys()

	if nb == 1:
		val = [random.choice(l) for l in dic.values()]

		if ev_oth == True :	
			for (a,b) in zip(dic_keys[::2], val[::2]):
				im[a][b] = random.uniform(thr_new[0],thr_new[1])
		else:
			for (a,b) in zip(dic_keys, val):
				im[a][b] = random.uniform(thr_new[0],thr_new[1])

	if nb > 1:
		val = []
		for l in dic.values():
			if nb >= len(l):
				val.append([random.sample(l, len(l))])
			else:
				val.append([random.sample(l, nb)])

		if ev_oth == True :	
			for (a,b) in zip(dic_keys[::2], val[::2]):
				for c in b:
					im[a][c] = random.uniform(thr_new[0],thr_new[1])
		else:
			for (a,b) in zip(dic_keys, val):
				for c in b:
					im[a][b] = random.uniform(thr_new[0],thr_new[1])

	return im


def pix_mod_run(thr_cut, thr_new, nb, ev_oth, tot_im, dump, dataset=mnist.load_data()):
	(x, y), (x_test, y_test) = dataset

	start = time.time()
	print("\nModifying train images...")
	mod_x = [pixel_mod(x_i, thr_cut, thr_new, nb, ev_oth) for x_i in x[0:tot_im]]
	mod_y = y[0:tot_im]
	print("Done\n")
	print("Modifying test images...")
	mod_x_test = [pixel_mod(x_i, thr_cut, thr_new, nb, ev_oth) for x_i in x_test[0:tot_im]]
	mod_y_test = y_test[0:tot_im]
	print("Done\n")

	if dump == True:
		name = str(thr_cut) + "_" + str(thr_new) + "_" + str(nb) + "_" + str(ev_oth) + "_" + str(tot_im)
		if not os.path.exists("./data/"):
			os.makedirs("data")
		print("Dumping into file...")
		pickle.dump( ((mod_x, mod_y), (mod_x_test, mod_y_test)), open( "./data/mnist_pix_mod_" + name + ".p", "wb" ))
		print("Done\n")

	end = time.time()
	print("Run time: " + str(round(end - start, 3)) + "s")

	return ((mod_x, mod_y),(mod_x_test, mod_y_test))