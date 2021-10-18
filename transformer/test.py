"""
@创建日期 ：2021/10/18
@修改日期 ：2021/10/18
@作者 ：jzj
@功能 ：
"""

import numpy as np
import pathlib

res = [1, 2, 3, 4, 5, 6]


with open("/Users/jiangzhenjie/Desktop/test.txt", "w") as f:
    for i in res:
        print(i, file=f)