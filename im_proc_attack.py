from matplotlib import pyplot as plt
from collections import defaultdict
# from keras.datasets import mnist
import numpy as np
import pickle
import random
import time

from pixel_mod import pix_mod_run
from band_gen import band_gen_run
from gauss_gen import gauss_gen

#-------------------------------------------------- Pix Mod ---------------------------------------------------#

# thr_cut = 200
# thr_new = (0,10)
# nb = 3
# ev_oth = False
# tot_im = 5000
# dump = False

# mod = pix_mod_run(thr_cut, thr_new, nb, ev_oth, tot_im, dump)

#-------------------------------------------------- Band Gen ---------------------------------------------------#

# start_pos = 13
# end_pos = 15
# new_val = (200,256)
# same = False
# tot_im = 5000
# dump = False

# mod = band_gen_run(start_pos, end_pos, new_val, same, tot_im, dump)

#-------------------------------------------------- Gauss Gen --------------------------------------------------#

sig = 1
tot_im = 5000
dump = False

mod = gauss_gen(sig, tot_im, dump)

#---------------------------------------------------- Plot -----------------------------------------------------#
(x, y), (x_test, y_test) = mod

im = 70

plt.imshow(x[im], cmap='gray')
plt.show()

print(y[70])

plt.imshow(x_test[im], cmap='gray')
plt.show()

print(y_test[70])
