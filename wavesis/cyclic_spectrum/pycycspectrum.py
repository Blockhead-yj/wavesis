# -*- encoding: utf-8 -*-
'''
File: -> pycycspectrum.py
Author: DYJ
Date: 2023/02/22 11:34:01
Desc: python实现的循环谱相关函数CPS_W、SCoh_W
version: 1.0
'''

import cupy as cp
from cupy.fft import fft
import numpy as np
from skimage.util import view_as_windows

def CPS_W_cupy(y, x, alpha, nfft=None, Noverlap=None, Nwindow=256, opt='sym'):
    """
    The python implement of CPS_W in matlab cyclostationary_toolbox
        Welch's estimate of the (Cross) Cyclic Power Spectrum 
        of signals y and x at cyclic frequency alpha:  

    Parameters
    ----------
    y, x : 1D-array
        data(use analytic signal to avoid correlation between + and - frequencies)
    alpha : float
        cyclic frequency (normalized by sampling frequency), in [0,1]
    nfft : int, optional
        FFT length, by default 2*Nwind
    Noverlap : int, optional
        block overlap, by default Noverlap = 2/3*Nwind with a hanning window 
                            or Noverlap = 1/2*Nwind with a half-sine window
    Nwindow : int|1D-array, optional
        window length or window, by default 256
    opt : Literal[sym, aym], optional
        the calculation version of CPS, by default 'sym'
    """ 
    # 获取hanning窗
    if isinstance(Nwindow, (int,)):
        window = cp.hanning(Nwindow)
    else:
        window = Nwindow[:]
        Nwindow = len(window)
    
    # 检查输入参数
    nfft = 2 * Nwindow if nfft is None else nfft
    Noverlap = int(np.floor(2 / 3 * Nwindow)) if Noverlap is None else int(np.floor(Noverlap))
    assert 0 <= alpha <= 1, 'alpha must be in [0,1] !!'
    assert Nwindow > Noverlap, 'Window length must be > Noverlap'
    assert nfft >= Nwindow, 'Window length must be <= nfft'
    
    # number of windows
    n = len(x)
    K = cp.floor((n - Noverlap) / (Nwindow - Noverlap))
    # 计算CPS
    freq = np.arange(nfft) / nfft
    t = np.arange(n)
    if opt=='sym':
        y = y * np.exp(-1j * np.pi * alpha * t)
        x = x * np.exp(1j * np.pi * alpha * t)
    else:
        x = x * np.exp(2j * np.pi * alpha * t)
    
    splited_x = view_as_windows(x, Nwindow, Nwindow - Noverlap)
    splited_y = view_as_windows(y, Nwindow, Nwindow - Noverlap)
    del x
    del y
    # 将array转化为cupy array， 加快运算速度
    splited_x = cp.array(splited_x, dtype=cp.complex64)
    splited_y = cp.array(splited_y, dtype=cp.complex64)
    
    splited_x *= window
    splited_y *= window
    
    fft_x = fft(splited_x, nfft, axis=1)
    fft_y = fft(splited_y, nfft, axis=1)
    
    CPS = fft_y * cp.conj(fft_x)
    CPS = cp.sum(CPS, axis=0)
    
    # 归一化
    KMU = K * cp.linalg.norm(window, 2) ** 2  # Normalizing scale factor
    CPS /= KMU
    
    return {'S':CPS.get(), 'f':freq, 'K':K.get()}

def CPS_W(y, x, alpha, nfft=None, Noverlap=None, Nwindow=256, opt='sym'):
    """
    (Numpy Version)The python implement of CPS_W in matlab cyclostationary_toolbox
        Welch's estimate of the (Cross) Cyclic Power Spectrum 
        of signals y and x at cyclic frequency alpha:  

    Parameters
    ----------
    y, x : 1D-array
        data(use analytic signal to avoid correlation between + and - frequencies)
    alpha : float
        cyclic frequency (normalized by sampling frequency), in [0,1]
    nfft : int, optional
        FFT length, by default 2*Nwind
    Noverlap : int, optional
        block overlap, by default Noverlap = 2/3*Nwind with a hanning window 
                            or Noverlap = 1/2*Nwind with a half-sine window
    Nwindow : int|1D-array, optional
        window length or window, by default 256
    opt : Literal[sym, aym], optional
        the calculation version of CPS, by default 'sym'
    """ 
    # 获取hanning窗
    if isinstance(Nwindow, (int,)):
        window = np.hanning(Nwindow)
    else:
        window = Nwindow[:]
        Nwindow = len(window)

    # 检查输入参数
    nfft = 2 * Nwindow if nfft is None else nfft
    Noverlap = int(np.floor(2 / 3 * Nwindow)) if Noverlap is None else Noverlap
    assert 0 <= alpha <= 1, 'alpha must be in [0,1] !!'
    assert Nwindow > Noverlap, 'Window length must be > Noverlap'
    assert nfft >= Nwindow, 'Window length must be <= nfft'
    
    # number of windows
    n = len(x)
    K = np.floor((n - Noverlap) / (Nwindow - Noverlap))
    # 计算CPS
    freq = np.arange(nfft) / nfft
    t = np.arange(n)
    if opt=='sym':
        y = y * np.exp(-1j * np.pi * alpha * t)
        x = x * np.exp(1j * np.pi * alpha * t)
    else:
        x = x * np.exp(2j * np.pi * alpha * t)
    
    splited_x = view_as_windows(x, Nwindow, Nwindow - Noverlap)
    splited_y = view_as_windows(y, Nwindow, Nwindow - Noverlap)
    del x
    del y
    # 如果不进行这一步，两个版本的结果会不一致，但是不知道为什么这一步会导致结果差异
    splited_x = np.array(splited_x, dtype=np.complex64)
    splited_y = np.array(splited_y, dtype=np.complex64)
    
    splited_x *= window
    splited_y *= window
    
    fft_x = np.fft.fft(splited_x, nfft, axis=1)
    fft_y = np.fft.fft(splited_y, nfft, axis=1)
    
    CPS = fft_y * np.conj(fft_x)
    CPS = np.sum(CPS, axis=0)
    
    # 归一化
    KMU = K * np.linalg.norm(window, 2) ** 2  # Normalizing scale factor
    CPS /= KMU
    
    return {'S':CPS, 'f':freq, 'K':K}

def SCoh_W(y, x, alpha, nfft=None, Noverlap=None, Nwindow=256, opt='sym', cupy=False):
    """
    The python implement of SCoh_W in matlab cyclostationary_toolbox
        Welch's estimate of the Cyclic Spectral Coherence 
        of signals y and x at cyclic frequency alpha:  

    Parameters
    ----------
    y, x : 1D-array
        data(use analytic signal to avoid correlation between + and - frequencies)
    alpha : float
        cyclic frequency (normalized by sampling frequency), in [0,1]
    nfft : int, optional
        FFT length, by default 2*Nwind
    Noverlap : int, optional
        block overlap, by default Noverlap = 2/3*Nwind with a hanning window 
                            or Noverlap = 1/2*Nwind with a half-sine window
    Nwindow : int, optional
        window length, by default 256
    opt : Literal[sym, aym], optional
        the calculation version of CPS, by default 'sym'
    """     
    n = len(x)
    t = np.arange(n)
    window = cp.hanning(Nwindow)
    if opt=='sym':
        y = y * np.exp(-1j * np.pi * alpha * t)
        x = x * np.exp(1j * np.pi * alpha * t)
    else:
        x = x * np.exp(2j * np.pi * alpha * t)
    func = CPS_W_cupy if cupy else CPS_W
    Syx = func(y,x,0,nfft,Noverlap,window,opt)
    Sy = func(y,y,0,nfft,Noverlap,window,opt)
    Sx = func(x,x,0,nfft,Noverlap,window,opt)
    C = Syx['S'] / np.sqrt(Sx['S'] * Sy['S'])
    
    return {'C':C, 'f':Sx['f'], 'K':Sx['K'], 'Syx':Syx, 'Sx':Sx, 'Sy':Sy}

def CPS_W_parrallel(alpha, y, x, nfft=None, Noverlap=None, Nwindow=256, opt='sym'):
    return CPS_W_cupy(y, x, alpha, nfft, Noverlap, Nwindow, opt)

if __name__=='__main__':
    t = np.linspace(0,10,80000)
    signal = np.sin(50 * np.pi * t)
    
    print(CPS_W(signal, signal, 0.1, Nwindow=8000))