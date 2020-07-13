import random
import numpy as np
#y=aX^3+bX^2+cX^1+dX^0
def gen_2d_3exp_data(param_a, param_b, param_c, param_d, x_min, x_max, num, error_val):
    data = np.zeros((num, 3))
    for i in range(num):
        x = x_min + i*(x_max - x_min) / num
        err = error_val * (0.5-random.random())
        y = param_a*x**3 + param_b*x**2 + param_c*x + param_d
        data[i, :] = [x, y+err, y]

    return data