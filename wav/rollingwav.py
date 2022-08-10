# -*- encoding: utf-8 -*-
'''
Projects -> File: -> rollingwav.py
Author: DYJ
Date: 2022/07/19 14:27:01
Desc: 滑动窗类, 实现Wav类的窗划分计算和聚合
version: 1.0
'''

from inspect import isfunction
import numpy as np

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
        if not isfunction(getattr(self.Wav, name)):
            res = [getattr(i_wav, name) for i_wav in self.get_window_wav()]
        try:
            res_wav = self.__class__(res, self.frequency / self.step)
        except Exception:
            res_wav = res
        
        return res_wav

    # To Do: 滑动窗计算函数