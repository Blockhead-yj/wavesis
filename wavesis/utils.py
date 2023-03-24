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
from scipy import linalg, signal

# 互谱密度函数
def cross_spectral_density(x, y):
    return np.fft.fft(x, norm='ortho') * np.conj(np.fft.fft(y, norm='ortho'))

# 获取维纳滤波器冲激响应函数
def get_frequency_response(i, u, period_length):
    # 转化为array处理
    i = np.asarray(i).squeeze()
    u = np.asarray(u).squeeze()
    assert len(i.shape)==1 and len(u.shape)==1, 'only 1D i/u is accepted!'
    assert len(i)==len(u), 'the length of i and u must be equal!'
    # 按照单个波形周期进行分割
    n_periods = len(i) // period_length
    clipped_i = i[:n_periods * period_length]
    clipped_u = i[:n_periods * period_length]
    i = clipped_i.reshape(-1, period_length)
    u = clipped_u.reshape(-1, period_length)
    # 计算互谱密度函数
    S_uu = np.mean([cross_spectral_density(i_u, i_u) for i_u in u], axis=0)
    S_iu = np.mean([cross_spectral_density(i_i, i_u) for i_i,i_u in zip(i, u)], axis=0)
    # 获得冲激响应
    frequency_response = S_iu / S_uu
    return frequency_response, u

def nextpow2(x):
    return np.ceil(np.log2(x))

def lpc(x, p):
    """仿照matlab中的lpc函数实现的python版本lpc,目前仅支持一维向量

    Parameters
    ----------
    x : 1D-array
        数据
    p : int
        线性预测系数阶数

    Returns
    -------
    array
        线性预测系数
    """    
    # 计算自相关矩阵(这是matlab利用的算法，自相关矩阵算法多的我都看不懂了，- -|)
    X = np.fft.fft(x, int(np.power(2, nextpow2(2*len(x) - 1))))
    R = np.fft.ifft(np.abs(X)**2)
    R = R / len(x)
    # 求解线性预测系数
    p = int(p) # 保证切片为int类型
    a = linalg.solve_toeplitz(R[:p], -R[1:p+1].reshape(-1,1))
    # a = np.insert(a, 0, 1)
    # 如果输入是实数，则仅返回系数的实数部分
    a = a if np.any(np.iscomplex(x)) else np.real(a)
    a = a.squeeze()
    return a


def lpc_filter(x, p):
    """lpc滤波(去除工频)

    Parameters
    ----------
    x : 1D-array
        数据
    p : int
        线性预测系数阶数

    Returns
    -------
    array
        残差信号
    """    
    a = lpc(x, p)
    est_x = signal.lfilter(np.concatenate([[0], -a]), 1, x)
    e = x - est_x
    return e

if __name__ == '__main__':
    x = np.array([2,6,8,9,5,6,1,4,3,8,4])
    y = np.array([9,4,5,8,6,7,9,6,4,7,1])
    print(lpc(y, 4))
    print(lpc_filter(x, 6))