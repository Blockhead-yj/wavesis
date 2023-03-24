# -*- encoding: utf-8 -*-
'''
Projects -> File: -> threephasewav.py
Author: DYJ
Date: 2022/07/27 16:06:49
Desc: 三相电类
version: 1.0
'''


from inspect import ismethod
from typing import List
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from scipy import signal # type: ignore

from .basewav import BaseWav
from .rollingwav import RollingWav, RollingWavBundle
from . import frequencydomainwav as fdwav
from . import timedomainwav as tdwav
from .utils import get_frequency_response
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
        self.width :List[int]|int = [len(i_wav) for i_wav in self.wavs]
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

    # 对称分量法
    def symmetrical_components(self, calc_voltage=False):
        """ 计算给定三相电力系统的对称分量

        Parameters
        ----------
        calc_voltage: bool
            是否计算电压的对称分量，默认为False，代表仅计算电流的对称分量；若为True，则仅计算电压的对称分量

        Returns
        -------
        symmetrical_componnents : DataFrame
            返回含有各相零序、正序和负序分量的df, columns=[A0, A1, A2, B0, B1, B2, C0, C1, C2]
        """
        a = np.exp((2 / 3) * np.pi * 1j) # 旋转算子a
        # 将信号转化为相量
        if not calc_voltage:
            complexor_a = self.data['ia'].hilbert(convert2real=False).values
            complexor_b = self.data['ib'].hilbert(convert2real=False).values
            complexor_c = self.data['ic'].hilbert(convert2real=False).values
        else:
            complexor_a = self.data['ua'].hilbert(convert2real=False).values
            complexor_b = self.data['ub'].hilbert(convert2real=False).values
            complexor_c = self.data['uc'].hilbert(convert2real=False).values
        symmetrical_components = pd.DataFrame()
        symmetrical_components['A0'] = (complexor_a + complexor_b + complexor_c)
        symmetrical_components['A1'] = (complexor_a + a * complexor_b + a ** 2 * complexor_c)
        symmetrical_components['A2'] = (complexor_a + a ** 2 * complexor_b + a * complexor_c)
        symmetrical_components['B0'] = symmetrical_components['A0']
        symmetrical_components['B1'] = symmetrical_components['A1'] / a
        symmetrical_components['B2'] = symmetrical_components['A2'] / (a ** 2)
        symmetrical_components['C0'] = symmetrical_components['A0']
        symmetrical_components['C1'] = symmetrical_components['A1'] / (a ** 2)
        symmetrical_components['C2'] = symmetrical_components['A2'] / a
        return symmetrical_components

    # 维纳滤波器去除工频及其谐波信号（《Electrical signals analysis of an asynchronous motor for bearing fault detection》）
    def wiener_filter(self):
        '''
        基于Ibrahim的《Electrical signals analysis of an asynchronous motor for bearing fault detection》文章，
        以电压信号作为参考信号构建维纳滤波器，滤除电流信号中的电气信号，保留机械信号。
        输出为 (重构信号， 残差， 原始电流信号)
        '''
        # 检查电压电流信号是否齐全
        if self._data_check() != '电流电压数据完整':
            raise Exception('电压电流数据不全，无法进行滤波！')
        # 使用A相电流与电压
        i = self.data['ia'].values
        u = self.data['ua'].values
        # 获取冲激响应函数
        power_freq, _ = self.data['ia'].baseband
        period_length = int(np.round(self.sample_frequency / power_freq))
        H, windowed_u = get_frequency_response(i, u, period_length)
        ifft_hat = np.hstack([np.fft.ifft(np.fft.fft(i_u) * H) for i_u in windowed_u])
        i_hat = ifft_hat.real
        residual = i - i_hat
        return i_hat, residual, i

    def plot(self, voltage=False):
        if voltage:
            plt.subplot(121)
            self.data['ia'].plot('r-', label='A相电流')
            self.data['ib'].plot('g-', label='B相电流')
            self.data['ic'].plot('b-', label='C相电流')
            plt.legend()
            plt.subplot(122)
            self.data['ua'].plot('r-', label='A相电压')
            self.data['ub'].plot('g-', label='B相电压')
            self.data['uc'].plot('b-', label='C相电压')
            plt.legend()
        else:
            self.data['ia'].plot('r-', label='A相电流')
            self.data['ib'].plot('g-', label='B相电流')
            self.data['ic'].plot('b-', label='C相电流')
            plt.legend() 
        return None
    

    # To do: 相位不平衡算法；幅值不平衡算法

if __name__ == '__main__':
    t = np.arange(0, 10, 0.01)
    phaseA = np.cos(2 * np.pi * t) * np.cos(100 * np.pi * t)
    phaseB = np.cos(2 * np.pi * t + np.pi * 2 / 3) * np.cos(100 * np.pi * t + np.pi * 2 / 3)
    phaseA_wav = tdwav.TimeDomainWav(phaseA)
    phaseB_wav = tdwav.TimeDomainWav(phaseB)
    wav_bundle = WavBundle(A=phaseA_wav, B=phaseB_wav)

    wav_bundle.fft()