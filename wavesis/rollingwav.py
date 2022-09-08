# -*- encoding: utf-8 -*-
'''
Projects -> File: -> rollingwav.py
Author: DYJ
Date: 2022/07/19 14:27:01
Desc: 滑动窗类, 实现Wav类的窗划分计算和聚合
version: 1.0
'''

from inspect import ismethod
from pdb import set_trace
import numpy as np

# To do: 对于特别耗时的计算，改写成view_as_windows的版本，提高运算效率
# from skimage.util import view_as_windows


class RollingWav(object):
    '''
    滑动窗类, 实现Wav类的窗滑动计算和聚合
    '''
    def __init__(self, Wav, window_width, step=1) -> None:
        self.Wav = Wav
        self.window_width = window_width
        self.step = step
        if self.step == 1:
            self.total_n_step = int(self.Wav.length - self.window_width + 1)
        else :
            self.total_n_step = int(np.floor((self.Wav.length - self.window_width) / self.step) + 1)

    def get_window_wav(self):
        for i_step in range(self.total_n_step):
            if self.Wav[i_step * self.step : i_step * self.step + self.window_width] is not None:
                yield self.Wav[i_step * self.step : i_step * self.step + self.window_width]
            else:
                raise StopIteration

    def __getattr__(self, name):
        window_wav = self.get_window_wav()
        res = [getattr(i_wav, name) for i_wav in window_wav]
        if not ismethod(getattr(self.Wav, name)): # 对于函数属性，直接存储为结果
            try:
                res_wav = self.Wav.__class__(res, self.Wav.frequency / self.step)
            except Exception:
                res_wav = res
        else: # 对于函数方法，先保存为RollingWav的一个属性，再随后对RollingWav的调用中，再将方法参数输入
            self.funcs = res
            res_wav = self
        return res_wav
    
    # 对于函数方法的rolling调用，参数在这里输入
    # 由于函数方法的输出多种多样，因此这里的输出不强制转换为Wav，而是直接输出列表
    def __call__(self, *args, **kwargs):
        res = [func(*args, **kwargs) for func in self.funcs] 
        return res

    def apply(self, func, *args, **kwargs):
        window_wav = self.get_window_wav()
        res = [func(wav.values, *args, **kwargs) for wav in window_wav]
        return res

class RollingWavBundle(object):
    '''
    滑动窗类, 实现WavBundle类的窗滑动计算和聚合
    WavBundle的基础运算方式，是对Bundle里面的每一个Wav进行相同的运算
        对于Wav的属性调用，直接用.调用
        对于Wav的函数调用，统一使用apply函数
    WavBundle的高级运算是需要多个波结合起来运算，比如三相电的park矢量变换和对称分量法
        这类计算统一直接使用WavBundle的函数调用，即使用.调用
    '''
    def __init__(self, WavBundle, window_width, step=1) -> None:
        self.WavBundle = WavBundle
        self.window_width = window_width
        self.step = step

    def get_window_wavbundle(self):
        if isinstance(self.WavBundle.shape[1], (list, tuple)):
            wav_length = min(self.WavBundle.shape[1])
        else:
            wav_length = self.WavBundle.shape[1]
        if self.step == 1:
            self.total_n_step = int(wav_length - self.window_width + 1) # 如果Bundle里面的Wav长度各不相同，去最短的长度进行切分
        else :
            self.total_n_step = int(np.floor((wav_length - self.window_width) / self.step) + 1)
        for i_step in range(self.total_n_step):
            if self.WavBundle[i_step * self.step : i_step * self.step + self.window_width] is not None:
                yield self.WavBundle[i_step * self.step : i_step * self.step + self.window_width]
            else:
                raise StopIteration

    def __getattr__(self, name):
        # To do： 思考如何把这一部分跟rollingwav的统一起来，应该是可以统一的
        if ismethod(getattr(self.WavBundle, name)): # 对于函数方法，将先保存在self.funcs里，在随后的__call__中传入参数并调用
            window_wavbundle = self.get_window_wavbundle()
            self.funcs = [getattr(i_wavbundle, name) for i_wavbundle in window_wavbundle]
            return self
        else: # 对于函数属性，直接调用
            res = [getattr(i_wav.rolling(self.window_width, self.step), name) for i_wav in self.WavBundle.wavs]
            return self.WavBundle.__class__(**dict(zip(self.WavBundle.wavnames, res)))   

    # 对于函数方法的rolling调用，参数在这里输入
    # 由于函数方法的输出多种多样，因此这里的输出不强制转换为Wav，而是直接输出列表
    def __call__(self, *args, **kwargs):
        res = [func(*args, **kwargs) for func in self.funcs] 
        return res

    def apply(self, func, *args, **kwargs):
        window_wavbundle = self.get_window_wavbundle()
        res = [i_wavbundle.apply(func, *args, **kwargs) for i_wavbundle in window_wavbundle]
        # return RollingWavBundle(**dict(zip(self.WavBundle.wavnames, res)))
        return res