# -*- encoding: utf-8 -*-
'''
Projects -> File: wavesis -> basewav.py
Author: DYJ
Date: 2022/07/19 12:14:09
Desc: 基础波类/基础振动信号类 
      该类实现所有不依赖时间信息或频率信息的指标计算和变换      
      注: 此处的振动是指自然界中广泛存在的物质系统的一种普遍运动形式, 通常指一个物理量在某一平衡值附近随时间而往复变化的过程 ———— 徐平《振动信号处理与数据分析》
version: 1.0
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .rollingwav import RollingWav


# Base Class
class BaseWav(object):
    '''
    基础波类，基础振动信号类
    '''
    def __init__(self, values, frequency=None):
        """ 
        初始化Wav

        Parameters
        ----------
        values : 1D list or array-like
        值序列/信号序列
        frequency: int or 1D list
        频率, 可选参数

        Returns
        -------
        None
        """
        self.values = np.asarray(values).squeeze()
        # 检查初始化数据
        if len(self.values.shape) > 1:
            raise Exception('Wave data can not have more than 1-dimension!') # todo： 考虑要不要把这个改成装饰器
        self.length = len(self.values)
        self.frequency = frequency

    # 直接打印时，显示振动信号的值
    def __repr__(self) -> str:
        return str(self.values)

    # 实现切片功能，对Wav类进行切片仍然得到Wav类
    def __getitem__(self, item): 
        cls = type(self) 
        if isinstance(item, slice):
            if self.frequency is None or isinstance(self.frequency, (int, float)):
                return self.__class__(self.values[item], self.frequency)
            elif len(self.frequency) > 1:
                return self.__class__(self.values[item], self.frequency[item])
            else :
                Warning('The type of self.frequency is incorrect, please check it!')
                return self.__class__(self.values[item], self.frequency)
        elif isinstance(item, int):
            return self.values[item]

    def __len__(self):
        return self.length

    # To do: 实现基本的加减乘除功能
    def _check_identity(self, other):
        if self.__class__ != other.__class__:
            return 'Different Domain'
        elif self.frequency != other.frequency:
            return 'Different Frequency'
        elif self.length != other.length:
            return 'Different Length'
        else:
            return 'ok'

    def __add__(self, other):
        if self._check_identity(other) == 'ok':
            return self.__class__(self.values + other.values, self.frequency)
        else:
            raise Exception('Two wavs with ' + self._check_identity(other) + ' can not be added!')
    
    def __sub__(self, other):
        if self._check_identity(other) == 'ok':
            return self.__class__(self.values - other.values, self.frequency)
        else:
            raise Exception('Two wavs with ' + self._check_identity(other) + ' can not be subtracted!')

    def __mul__(self, other):
        if self._check_identity(other) == 'ok':
            return self.__class__(self.values * other.values, self.frequency)
        elif isinstance(other, (int, float)):
            return self.__class__(self.values * other, self.frequency)
        else:
            raise Exception('Two wavs with ' + self._check_identity(other) + ' can not be multiplied!')

    def __div__(self, other):
        pass
    
    # utils
    # 零均值化（去均值化）
    def demean(self, inplace=False):
        if inplace:
            self.values = self.values - np.mean(self.values)
        else:
            return self.__class__(self.values - np.mean(self.values), self.frequency)
    # 归一化
    def normalize(self, inplace=False):
        if inplace:
            self.values = self.values / np.max(np.abs(self.values))
        else:
            return self.__class__(self.values / np.max(np.abs(self.values)), self.frequency)        

    # properties
    # 不依赖任何时间信息或频率信息计算的常用指标
    # 指标全部以属性而不是函数的形式调用
    @property
    def PP(self):
        '''
        最大最小值, 峰峰值PP
        '''
        return np.max(self.values) - np.min(self.values)

    @property
    def Mean(self):
        '''
        平均值(一阶矩)
        '''
        return np.mean(self.values)

    @property
    def RMS(self):
        '''
        均方根值(二阶矩)
        '''
        return np.sqrt(np.mean(np.power(self.values, 2)))

    @property
    def Var(self):
        '''
        方差
        '''
        return np.mean(np.power(self.values - self.Mean, 2))

    @property
    def SD(self):
        '''
        标准差
        '''
        return np.sqrt(self.Var)

    @property
    def Skewness(self):
        '''
        偏度 SK(三阶矩)
        '''
        return np.mean(np.power(self.values - self.Mean, 3)) / np.power(self.SD, 3)
    
    @property
    def IM(self):
        '''
        脉冲指标 IM(Impulse Indicator)
        IM 是信号绝对值的最大值与绝对平均值之商。信号中的冲击越大, IM 越大。
        '''
        return np.max(np.abs(self.values)) / np.mean(np.abs(self.values))

    @property
    def CR(self):
        '''
        峰值指标 CR(Crest Indicator/Factor)
        '''
        return np.max(np.abs(self.values)) / self.RMS

    @property
    def CL(self):
        '''
        裕度指标 CL(Clearance Indicator)
        '''
        return np.max(np.abs(self.values)) / np.power(np.mean(np.sqrt(np.abs(self.values))), 2)

    @property
    def Kurtosis(self):
        '''
        峰度 KU(四阶矩)
        '''
        return np.mean(np.power(self.values - self.Mean, 4)) / np.power(self.SD, 4) - 3

    # 不依赖任何时间和频率信息进行的变换
    def hilbert(self, convert2real=True):
        """
        进行希尔伯特变换

        Parameters  :
        ----------
        convert2real: bool
        是否要将傅里叶变换的结果变为实数(取模)

        Returns  :
        -------
        hilbert_wav: Wav
        经过希尔伯特变换后的波
        """
        if convert2real:
            hilbert = np.abs(signal.hilbert(self.values))
        else:
            hilbert = signal.hilbert(self.values)

        hilbert_wav = self.__class__(hilbert, self.frequency)

        return hilbert_wav

    # 基本绘图函数，后续可对子类定制自己的绘图函数
    def plot(self, *args, **kwargs):
        plt.plot(self.values, *args, **kwargs)
        return None

    # 实现滑动窗计算，每个滑动窗都是一个Wav，可以在每个滑动窗进行所有指标的计算和转换
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
        rolling_wav: RollingWav
        A generator of Wav
        '''
        rolling_wav = RollingWav(self, window_width, step)
        return rolling_wav
    # 拓展函数，可对数据直接进行自定义的变换和计算
    def apply(self, func, *args, **kwargs):
        self.values = func(self.values, *args, **kwargs)
        return self

if __name__ == "__main__":
    # init basewav
    x = np.linspace(0, 1000, 800000)
    y1 = np.cos(x) 
    y2 = 3 * np.sin(10 * x) + 1
    test_wav1 = BaseWav(y1)
    test_wav2 = BaseWav(y2)
    # test basic function
    print('test basic function'.center(40, '-'))
    print(test_wav1)
    part_wav = test_wav1[100:1000]

    wav_sum = test_wav1 + test_wav2
    wav_diff = test_wav1 - test_wav2
    wav_prod = test_wav1 * test_wav2

    # test utils
    print('test utils'.center(40, '-'))
    test_wav2.demean(inplace=True)

    # test properties
    print('test properties'.center(40, '-'))
    for property in ["PP", "Mean", "RMS", "Var", "SD", "Skewness", "IM", "CR", "CL", "Kurtosis"]:
        print(property +": %.4f" % getattr(wav_sum, property))

    # test tranformation & plot
    print('test transformation'.center(40, '-'))
    wav_prod[:800].plot('r-', label='raw data')
    wav_prod.hilbert()[:800].plot('b-', label='hilbert')
    plt.legend()
    plt.show()

    # test the rolling calculation
    print('test rolling calculation'.center(40, '-'))
    print(wav_sum.rolling(100, 50).RMS)


