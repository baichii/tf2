"""
@创建日期 ：2021/10/28
@修改日期 ：2021/10/28
@作者 ：jzj
@功能 ：
"""


import random
import numpy as np
# import tensorflow as tf


a = [0, 2, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35]

board = np.ones(36)

board[a] = 0
print(np.asarray(board).reshape(6, 6))
