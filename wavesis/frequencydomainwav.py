# -*- encoding: utf-8 -*-
'''
Projects -> File: -> frequencydomainwav.py
Author: DYJ
Date: 2022/07/19 14:44:53
Desc: 频域信号类
      在频域内进行计算的指标放在这个类里
version: 1.0
To do: 与目前系统里的计算方式对比，看结果是否有差异，看实现是否正确
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .basewav import BaseWav
from .rollingwav import RollingWav
from . import timedomainwav as tdwav


class FrequencyDomainWav(BaseWav):
    '''
    频域信号

    在频域内进行计算的指标放在这个类里
    '''
    def __init__(self, values, frequency):
        ''' 
        初始化FDWav, 频域信号初始化必须提供频率信息
        
        Parameters
        ----------
        values : 1D list or array-like
        信号的强度值
        frequency : int
        信号对应的频率

        Returns
        -------
        None
        '''
        self.values = np.asarray(values).squeeze()
        if len(self.values.shape) > 1:
            raise Exception('Wave data can not have more than 1-dimension!')
        self.length = len(self.values)
        self.frequency = frequency

    # utils
    def get_frequency_band(self, lower_bound, higher_bound):
        """
        获取指定频带的数据

        Parameters  :
        ----------
        lower_bound: int or float
        频带下限, 单位Hz
        end_freq: int or float
        频带上限, 单位Hz

        Returns  : 
        -------
        part_wav: FrequencyDomainWav
        指定频带的数据
        
        """
        part_index = (self.frequency >= lower_bound) & (self.frequency < higher_bound)
        part_wav = self.__class__(self.values[part_index], self.frequency[part_index])

        return part_wav

    def find_max_peak(self, start_freq=0, end_freq=np.inf):
        """
        在一定频率范围内查找最大的幅值值及相应的频率
        默认为从0到正无穷查找，这样找出来的是基频
        Parameters  :
        ----------
        start_freq: int or float
        起始频率范围
        end_freq: int or float
        终止频率范围

        Returns  : 
        -------
        (max_peak_frequency, max_peak_magnitude)
        
        """
        
        search_space = self.get_frequency_band(start_freq, end_freq)
        max_peak_frequency = search_space.frequency[np.argmax(search_space.values)]
        max_peak_magnitude = np.max(search_space.values)

        return (max_peak_frequency, max_peak_magnitude)
    # 在频域内计算的指标/依赖频率信息计算的指标
    @property
    def FC(self):
        '''
        重心频率
        '''
        return np.average(self.frequency, weights=self.values)

    @property
    def MSF(self):
        '''
        均方频率
        '''
        return np.average(np.power(self.frequency, 2), weights=self.values)

    @property
    def VF(self):
        '''
        频率方差
        '''
        square_deviate_from_mean = np.power(self.frequency - self.FC, 2)
        return np.average(square_deviate_from_mean, weights=self.values)

    @property
    def RMSF(self):
        '''
        均方根频率
        '''
        return np.sqrt(self.MSF)

    @property
    def RVF(self):
        '''
        频率标准差
        '''
        return np.sqrt(self.VF)
    
    @property
    def baseband(self):
        return self.find_max_peak()

    def THD(self, max_harmonic_num=150, leaky_index=1):
        """ 谐波畸变率
        由少青的代码改编而来

        Parameters
        ----------
        max_harmonic_num : int
            最大谐波数
        leaky_index : int
            泄露指数，在谐波附近多大频率范围内寻找谐波峰值
        Returns
        -------
        total_harmonic_distortion : float
            总谐波畸变率

        """
        # 计算基频幅值
        fundamental_frequency = self.frequency[np.argmax(self.values)]
        fundamental_magnitude = np.max(self.values)
        # 计算谐波含有率
        harmonic_component = 0
        # 计算可取的最大谐波数
        available_max_harmonic_num = np.max(self.frequency) // fundamental_frequency - 1
        max_harmonic_num = min(available_max_harmonic_num, max_harmonic_num)
        try :
            for multi in range(2, int(max_harmonic_num) + 1):
                # 指定谐波的搜寻范围
                frequency_series = [multi * fundamental_frequency - leaky_index,
                                    multi * fundamental_frequency + leaky_index]                                
                get_use_fft_data = self.values[(frequency_series[0] < self.frequency) &
                                                (self.frequency < frequency_series[1])]     
                get_use_mag = np.max(get_use_fft_data)
                max_magnitude_square = np.power(get_use_mag, 2)
                harmonic_component += max_magnitude_square
        except ValueError:
            print('The argument max_harmonic_num is too big!')
                
        square_harmonic_content = np.sqrt(harmonic_component)
        total_harmonic_distortion = square_harmonic_content / fundamental_magnitude
        return total_harmonic_distortion

    # 定制绘图函数
    def plot(self, magnitude_to_log=False, *args, **kwargs):
        if magnitude_to_log:
            plt.plot(self.frequency, np.log(self.values), *args, **kwargs)
        else:
            plt.plot(self.frequency, self.values, *args, **kwargs)
        
        return None

    # 在频域内进行的计算或变换
    def fft(self, window='hann', convert2real=True, demean=False):
        '''
        在频域内进行加窗傅里叶变换, 从频域到时域不需要除以数据长度

        Parameters
        ----------
        window : str
            加窗计算所使用的窗的名称, 支持scipy.signal.get_window内的所有窗
        convert2real : bool
            是否将傅里叶变换的结果从复数转化为实数(计算模)
        Returns
        -------
        TDWav
            傅里叶变换后的时域信号
        '''
        if demean:
            windowed_value = (self.values - self.Mean) * signal.get_window(window, self.length)
        # 加窗抑制频谱泄露
        windowed_value = self.values * signal.get_window(window, self.length)
        if convert2real:
            windowed_value_fft = np.abs(np.fft.fft(windowed_value)) 
        else:
            windowed_value_fft = np.fft.fft(windowed_value)

        return tdwav.TimeDomainWav(windowed_value_fft, sample_frequency=1)

# todo: 测试代码
if __name__ == "__main__":
    x = np.linspace(0, 1000, 10000)
    y = np.cos(x) 
    test_wav = FrequencyDomainWav(y, x)
    for property in ["FC", "MSF", "VF", "RMSF", "RVF"]:
        print(property +": %.4f" % getattr(test_wav, property))