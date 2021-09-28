"""
@创建日期 ：2021/9/27
@修改日期 ：2021/9/27
@作者 ：jzj
@功能 ：https://github.com/soosten/candy-canes/blob/main/agent.py
"""

import numpy as np

rng = np.random.default_rng(42)


supports = np.tile(np.arange(101.), (2, 100, 1))
beliefs = np.full_like(supports, 1/101.)

