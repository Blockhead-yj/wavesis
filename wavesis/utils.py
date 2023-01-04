# -*- encoding: utf-8 -*-
'''
Projects -> wavsis
File: -> utils.py
Author: DYJ
Date: 2023/01/03 16:16:36
Desc: 振动分析/信号处理工具函数
version: 1.0
'''

import numpy as np
from skimage.util import view_as_windows

# 互谱密度函数
def cross_spectral_density(x, y):
    return np.fft.fft(x, norm='ortho') * np.conj(np.fft.fft(y, norm='ortho'))

# 获取维纳滤波器冲激响应函数
def get_frequency_response(i, u, period_length):
    # 按照单个波形周期进行分割
    i = view_as_windows(np.asarray(i), period_length, period_length)
    u = view_as_windows(np.asarray(u), period_length, period_length)
    # 计算互谱密度函数
    S_uu = np.mean([cross_spectral_density(i_u, i_u) for i_u in u], axis=0)
    S_iu = np.mean([cross_spectral_density(i_i, i_u) for i_i,i_u in zip(i, u)], axis=0)
    # 获得冲激响应
    frequency_response = S_iu / S_uu
    return frequency_response, u