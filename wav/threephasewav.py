# -*- encoding: utf-8 -*-
'''
Projects -> File: -> threephasewav.py
Author: DYJ
Date: 2022/07/27 16:06:49
Desc: 三相电类
version: 1.0
'''

import warnings
import copy
from inspect import ismethod
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .basewav import BaseWav
from . import frequencydomainwav as fdwav
from . import timedomainwav as tdwav


class WavBundle(object):
    '''
    波束类
    把多个波“捆”成束进行统一处理，即对多个波进行统一的操作
    '''
    def __init__(self, **kwargs) -> None:
        self.wavnames = [key for key in kwargs.keys()]
        self.wavs = [copy.deepcopy(value) for value in kwargs.values()]

    def __getattr__(self, name):
        res = [getattr(i_wav, name) for i_wav in self.wavs]
        if not ismethod(getattr(self.wavs[0], name)): # 对于函数属性，直接存储为结果
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



class threephasewav(object):
    '''
    三相电类
    '''
    def __init__(self, phaseA_current=None, phaseB_current=None, phaseC_current=None, 
                    phaseA_voltage=None, phaseB_voltage=None, phaseC_voltage=None, 
                    wiring_structure=None) -> None:
        """
        初始化三相电类
        Parameters  :
        ----------
        phase*_current: list or 1D-array
        相电流
        phase*_voltage: list or 1D-array
        相电压
        wiring_structure: str
        接线方式："Y"(星形接线)或者"N"(三角接线)

        Returns  :
        -------
        None
        """
        self.phaseA_current = phaseA_current
        self.phaseB_current = phaseB_current
        self.phaseC_current = phaseC_current
        self.phaseA_voltage = phaseA_voltage
        self.phaseB_voltage = phaseB_voltage
        self.phaseC_voltage = phaseC_voltage
        self.wiring_structure = wiring_structure
        self.data_check = self._data_check()
        if self.data_check == '数据不完整':
            raise Exception('电压电流数据均不完整，无法进行后续计算!')
    
    def _data_check(self):
        # 检查电流数据是否完整
        if self.phaseA_current and self.phaseB_current and self.phaseC_current:
            current_ok = True
        else:
            current_ok = False
        # 检查电压数据是否完整
        if self.phaseA_voltage and self.phaseB_voltage and self.phaseC_voltage:
            voltage_ok = True
        else:
            voltage_ok = False
        if current_ok and voltage_ok:
            res = '电流电压数据完整'
        elif current_ok:
            res = '电流数据完整'
        elif voltage_ok:
            res = '电压数据完整'
        else:
            res = '数据不完整'
        return res

    # To do: 对称分量法
    # To do: 相位不平衡算法；幅值不平衡算法
