# -*- encoding: utf-8 -*-
'''
Projects -> File: -> rollingwav.py
Author: DYJ
Date: 2022/07/19 14:27:01
Desc: 滑动窗类, 实现Wav类的窗划分计算和聚合
version: 1.0
'''

from inspect import ismethod
import numpy as np
# To do: 对于特别耗时的计算，改写成view_as_windows的版本，提高运算效率
# from skimage.util import view_as_windows


class RollingWav(object):
    '''
    滑动窗类, 实现Wav类的窗划分计算和聚合

    对于指标, 可以直接计算
    对于函数计算或变换, 需要在这个类里面再次实现
    '''
    def __init__(self, Wav, window_width, step=1) -> None:
        self.Wav = Wav
        self.window_width = window_width
        self.step = step
        if step == 1:
            self.total_n_step = int(self.Wav.length - self.window_width + 1)
        else :
            self.total_n_step = int(np.floor((self.Wav.length - self.window_width) / self.step) + 1)

    def get_window_wav(self):
        for i_step in range(self.total_n_step):
            yield self.Wav[i_step * self.step : i_step * self.step + self.window_width]

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

    # To Do: 滑动窗计算函数