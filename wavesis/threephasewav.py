# -*- encoding: utf-8 -*-
'''
Projects -> File: -> threephasewav.py
Author: DYJ
Date: 2022/07/27 16:06:49
Desc: 三相电类
version: 1.0
'''


from inspect import ismethod
import numpy as np
import matplotlib.pyplot as plt

from .basewav import BaseWav
from .rollingwav import RollingWav, RollingWavBundle
from . import frequencydomainwav as fdwav
from . import timedomainwav as tdwav

class WavBundle(object):
    '''
    波束类
    把多个波“捆”成束进行统一处理，即对多个波进行统一的操作
    WavBundle的基础运算方式，是对Bundle里面的每一个Wav进行相同的运算
        对于Wav的属性调用，直接用.调用
        对于Wav的函数调用，统一使用apply函数
    WavBundle的高级运算是需要多个波结合起来运算，比如三相电的park矢量变换和对称分量法
        这类计算统一直接使用WavBundle的函数调用，即使用.调用
    '''
    def __init__(self, **kwargs) -> None:
        '''
        WavBundle的初始化按照WavName=Wav的方式任意长度输入
        WavName即为波的名称，如'A相电流'
        Wav应该是【BaseWav, fdwav.FrequencyDomainWav, tdwav.TimeDomainWav, RollingWav】中的任意一种
        '''
        # check data
        for k,v in kwargs.items():
            assert isinstance(v, (BaseWav, fdwav.FrequencyDomainWav, tdwav.TimeDomainWav, RollingWav)), k + 'is not a Wav, please check it!'
        self.wavnames = [key for key in kwargs.keys()]
        self.wavs = [value for value in kwargs.values()]
        self.length = len(self.wavs)
        self.width = [len(i_wav) for i_wav in self.wavs]
        if len(set(self.width)) == 1:
            self.width = self.width[0]
    
    @staticmethod
    def init_unnamed_wavs_from_list(li):
        return WavBundle(**{'wav_' + str(i): i_wav for i, i_wav in enumerate(li)})

    @property
    def shape(self):
        return self.length, self.width

    def __getattr__(self, name):
        if not ismethod(getattr(self.wavs[0], name)): # 对于函数属性，直接存储为结果
            res = [getattr(i_wav, name) for i_wav in self.wavs]
        elif name in self.__dict__: # 对于函数方法，只能调用WavBundle自身的实现，否则引发错误
            res = getattr(self, name)
        else:
            raise ValueError(self.__class__.__name__ + 'dose not have function: ' + name + ', please use apply instead.')
        return res

    # 直接打印时，显示wavs
    def __repr__(self) -> str:
        res = [str(i_wav) for i_wav in self.wavs]
        return '\n'.join(res)

    # 切片功能
    def __getitem__(self, item): 
        cls = type(self)
        if isinstance(item, slice):
            res = [i_wav[item] for i_wav in self.wavs]
            return self.__class__(**dict(zip(self.wavnames, res)))
        elif isinstance(item, int):
            res = [i_wav[item] for i_wav in self.wavs]
            return res
        else:
            raise ValueError(str(item) + 'must be slice or int!')

    def apply(self, func, *args, **kwargs):
        res = [func(wav, *args, **kwargs) for wav in self.wavs]
        try:
            res = np.asarray(res) if isinstance(res[0], (tuple,)) else self.__class__(**dict(zip(self.wavnames, res)))
        finally:
            return res

    # 实现滑动窗计算，每个滑动窗都是一个WavBundle，可以在每个滑动窗进行所有指标的计算和转换
    def rolling(self, window_width, step=1):
        '''
        Parameters
        ----------
        window_width : int 
        滑动窗的宽度
        step : int
        滑动窗的步长

        Returns
        -------
        rolling_wavbundle: RollingWavBundle
        A generator of Wav
        '''
        rolling_wavbundle = RollingWavBundle(self, window_width, step)
        return rolling_wavbundle

class threephasewav(WavBundle):
    '''
    三相电类
    '''
    def __init__(self, ia=None, ib=None, ic=None, ua=None, ub=None, uc=None, 
                    sample_frequency=8000, wiring_structure=None) -> None:
        """
        初始化三相电类
        Parameters  :
        ----------
        ia, ib, ic: list or 1D-array
        相电流
        ua, uv, uc: list or 1D-array
        相电压
        wiring_structure: str
        接线方式："Y"(星形接线)或者"N"(三角接线)

        Returns  :
        -------
        None
        """
        self.data = {'ia':ia, 'ib':ib, 'ic':ic,
                     'ua':ua, 'ub':ub, 'uc':uc}
        self.sample_frequency = sample_frequency
        self.wiring_structure = wiring_structure
        # self.data_check = self._data_check()
        # if self.data_check == '数据不完整':
        #     raise Exception('电压电流数据均不完整，无法进行后续计算!')
        for k,v in self.data.items():
            if not isinstance(v, (BaseWav, RollingWav)) and not v is None:
                self.data[k] = tdwav.TimeDomainWav(v, self.sample_frequency)
        WavBundle.__init__(self, **{k:v for k,v in self.data.items() if v is not None})

    @staticmethod
    def init_wavs_from_rawdata(data, colname=['A相电流', 'B相电流', 'C相电流'], sample_frequency=8000, with_voltage=False, voltage_colname=['A相电压', 'B相电压', 'C相电压']):
        if with_voltage:
            currentA = tdwav.TimeDomainWav(data[colname[0]], sample_frequency=sample_frequency)
            currentB = tdwav.TimeDomainWav(data[colname[1]], sample_frequency=sample_frequency)
            currentC = tdwav.TimeDomainWav(data[colname[2]], sample_frequency=sample_frequency)
            voltageA = tdwav.TimeDomainWav(data[voltage_colname[0]], sample_frequency=sample_frequency)
            voltageB = tdwav.TimeDomainWav(data[voltage_colname[1]], sample_frequency=sample_frequency)
            voltageC = tdwav.TimeDomainWav(data[voltage_colname[2]], sample_frequency=sample_frequency)
            return threephasewav(ia=currentA, ib=currentB, ic=currentC, ua=voltageA, ub=voltageB, uc=voltageC)
        else:
            currentA = tdwav.TimeDomainWav(data[colname[0]], sample_frequency=sample_frequency)
            currentB = tdwav.TimeDomainWav(data[colname[1]], sample_frequency=sample_frequency)
            currentC = tdwav.TimeDomainWav(data[colname[2]], sample_frequency=sample_frequency)
            return threephasewav(ia=currentA, ib=currentB, ic=currentC)

    def _data_check(self):
        # 检查电流数据是否完整
        if self.data['ia'] and self.data['ib'] and self.data['ic']:
            current_ok = True
        else:
            current_ok = False
        # 检查电压数据是否完整
        if self.data['ua'] and self.data['ub'] and self.data['uc']:
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

    def park_transform(self, magnitude=False):
        """park矢量变换，将原始电信号转换为i_d，i_q

        Parameters
        ----------
        magnitude : bool
            是否取模

        Returns
        -------
        wavs : WavBundle(i_d, i_q)
            park矢量转换后的数据
        or wav: TimeDomainWav
            park矢量模

        """
        ia, ib, ic = np.asarray(self.data['ia']), np.array(self.data['ib']), np.array(self.data['ic'])
        i_d = np.sqrt(2 / 3) * ia - np.sqrt(1 / 6) * ib - np.sqrt(1 / 6) * ic
        i_q = np.sqrt(1 / 2) * ib - np.sqrt(1 / 2) * ic
        if magnitude:
            return tdwav.TimeDomainWav(np.sqrt(np.power(i_d, 2) + np.power(i_q, 2)), self.sample_frequency)
        else:
            return WavBundle(i_d=tdwav.TimeDomainWav(i_d, self.sample_frequency), i_q=tdwav.TimeDomainWav(i_q, self.sample_frequency))

    def plot(self):
        self.data['ia'].plot('r-', label='A相电流')
        self.data['ib'].plot('g-', label='B相电流')
        self.data['ic'].plot('b-', label='C相电流')
        plt.legend()
        return None
    # To do: 对称分量法
    # To do: 相位不平衡算法；幅值不平衡算法

if __name__ == '__main__':
    t = np.arange(0, 10, 0.01)
    phaseA = np.cos(2 * np.pi * t) * np.cos(100 * np.pi * t)
    phaseB = np.cos(2 * np.pi * t + np.pi * 2 / 3) * np.cos(100 * np.pi * t + np.pi * 2 / 3)
    phaseA_wav = tdwav.TimeDomainWav(phaseA)
    phaseB_wav = tdwav.TimeDomainWav(phaseB)
    wav_bundle = WavBundle(A=phaseA_wav, B=phaseB_wav)

    wav_bundle.fft()
