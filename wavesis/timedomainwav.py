# -*- encoding: utf-8 -*-
'''
Projects -> File: -> timedomainwav.py
Author: DYJ
Date: 2022/07/19 14:25:22
Desc: 时域信号类
      在时域内计算的指标放在这个类里面计算，对这个类调用频域内计算的指标会自动转化为对该类进行默认的加窗傅里叶变换后再计算
version: 1.0
To do: 与目前系统里的计算方式对比，看结果是否有差异，看实现是否正确
'''


import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PyEMD import EMD

from .rollingwav import RollingWav

from .basewav import BaseWav
from . import frequencydomainwav as fdwav


class TimeDomainWav(BaseWav):
    '''
    时域信号

    在时域内计算的指标放在这个类里面计算，对这个类调用频域内计算的指标会自动转化为对该类进行默认的加窗傅里叶变换后再计算
    '''
    def __init__(self, values, sample_frequency):
        """ 
        初始化TDWav, 时域信号初始化必须提供采样率

        Parameters
        ----------
        values : 1D list or array-like
        Wav的值序列
        sample_frequency : int
        采样率, 单位: Hz

        Returns
        -------
        None
        """
        self.values = np.asarray(values).squeeze()
        if len(self.values.shape) > 1:
            raise Exception('Wave data can not have more than 1-dimension!')
        self.length = len(self.values)
        self.sample_frequency = sample_frequency
        self.time_length = self.length / self.sample_frequency

        # alias
        self.frequency = self.sample_frequency

    # 当需要计算的指标或进行的变换在时域内无法计算时，会自动转向该时域信号以默认参数进行的加窗傅里叶变换后的频域信号进行计算
    # 如果需要以非默认参数进行的傅里叶变换，请使用reset_freq_domain函数重置频域；或者直接在频域进行计算。
    def __getattr__(self, name):
        if name in ['FC', 'MSF', 'VF', 'RMSF', 'RVF']:  # 这些指标的计算应该在功率谱上进行计算
            return getattr(self.psd_spectrum, name)
        # lazy-evaluation, 惰性求值，知道被访问到时再进行计算而不是在初始化时就计算
        elif name == "freq_domain": 
            freq_domain = self.fft()
            setattr(self, name, freq_domain)
            return freq_domain
        elif name == "psd_spectrum":
            psd_spectrum = self.PSD_Periodogram()
            setattr(self, name, psd_spectrum)
            return psd_spectrum
        else:
            return getattr(self.freq_domain, name)

    def reset_freq_domain(self, window='hann', convert2real=True):
        '''
        重置频域
        如果不想使用默认的傅里叶变换参数的结果进行后续频域指标计算, 可以通过这个函数重新计算
        
        Parameters
        ----------
        window: str or tuple
        指定傅里叶变换时使用的窗函数, 该参数将被用于调用signal.get_window
        convert2real: bool
        是否要将傅里叶变换的结果变为实数(取模)
        
        Return
        ------
        None
        '''
        self.freq_domain = self.fft(window, convert2real)
        return None

    # 绘图函数
    def plot(self, *args, **kwargs):
        plt.plot(np.arange(self.length) / self.sample_frequency, self.values, *args, **kwargs)
        plt.xlabel('Time(s)')
        return None

    # 在时域上进行的变换
    def fft(self, window='hann', convert2real=True):
        '''
        在时域内进行加窗傅里叶变换，需要除以数据长度

        Parameters
        ----------
        window : str
            加窗计算所使用的窗的名称, 支持scipy.signal.get_window内的所有窗
        convert2real : bool
            是否将傅里叶变换的结果从复数转化为实数(计算模)
        Returns
        -------
        fft_wav: FrequencyDomainWav
            傅里叶变换后的频域信号
        '''
        # 加窗抑制频谱泄露
        window = signal.get_window(window, self.length)
        windowed_value = self.values * window
        adjust_coefficient = round(1 / np.mean(window), 4)
        if convert2real:
            fft_value = np.abs(np.fft.fft(windowed_value)) / self.length * adjust_coefficient
        else:
            fft_value = np.fft.fft(windowed_value) / self.length * adjust_coefficient
        
        # 计算频率
        freq = np.arange(self.length) / self.length * self.sample_frequency
        fft_wav = fdwav.FrequencyDomainWav(fft_value, freq)[:self.length // 2]  # 只取傅里叶变换结果的前一半
        return fft_wav

    # 频谱分析/功率谱分析/功率谱密度
    # 1. 周期图法/直接法
    def PSD_Periodogram(self, window='hann'):
        """ 
        直接法求功率谱密度
        直接法又称周期图法(Periodogram)

        Parameters
        ----------
        window: str or tuple
        指定傅里叶变换时使用的窗函数, 该参数将被用于调用signal.get_window

        Returns
        -------
        psd_spectrum : FrequencyDomainWav
        """

        freq, spectrum = signal.periodogram(self.values, self.sample_frequency, window)
        psd_spectrum = fdwav.FrequencyDomainWav(spectrum, freq)
        return psd_spectrum

    # 2. Welch法/改进的直接法(加窗)
    def PSD_Welch(self, window='hann', nperseg=8192):
        '''
        用Welch法计算功率谱密度 PSD(Power Spectral Density)
        Welch法是改进的直接法, 先将信号分段加窗后计算功率谱, 随后对每段数据的功率谱进行平均

        Parameters
        ----------
        window: str or tuple
        指定傅里叶变换时使用的窗函数, 该参数将被用于调用signal.get_window
        npserseg: int
        每个window中含有的数据点的个数/window的长度

        Returns
        -------
        psd_spectrum : FrequencyDomainWav
        '''

        freq, spectrum = signal.welch(self.values, self.sample_frequency, window, nperseg)
        psd_spectrum = fdwav.FrequencyDomainWav(spectrum, freq)
        return psd_spectrum

    # 3. 间接法, 自相关函数法
    def PSD_autocorrelation(self, window='hann'):
        '''
        用自相关法计算功率谱密度 PSD(Power Spectral Density)
        自相关法是先计算序列的自相关系数, 然后对自相关系数进行傅里叶变换来得到功率谱

        Parameters
        ----------
        window: str or tuple
        指定傅里叶变换时使用的窗函数, 该参数将被用于调用signal.get_window

        Returns
        -------
        psd_spectrum : FrequencyDomainWav
        '''
        # 计算自相关的前提是函数f(t)的均值m(t)为零，因此需要先将数据demean
        demean_dat = self.values - self.Mean
        xcorr = np.correlate(demean_dat, demean_dat, mode="same")
        # 加窗抑制频谱泄露
        windowed_xcorr = xcorr * signal.get_window(window, len(xcorr))
        spectrum = np.abs(np.fft.fft(windowed_xcorr)) / self.length
        
        # 计算频率
        freq = np.arange(self.length) * self.sample_frequency / (self.length)
        psd_spectrum = fdwav.FrequencyDomainWav(spectrum, freq)
        return psd_spectrum

    # 熵
    # 1. 近似熵(https://zhuanlan.zhihu.com/p/494761890/)
    def calc_approximate_entropy(self, pattern_dimension=2, similarity_tolerance=0.20):
            """
            计算数据的近似熵

            Parameters  :
            ----------
            pattern_dimension: int
            模式维数
            similarity_tolerance: float
            相似容限阈值

            Returns  :
            -------
            approximate_entropy: float
            近似熵
            """
            if self.length > 5000 or self.length < 200:
                warnings.warn(f'The length of data used for approximate entropy should be 200~5000, not {self.length}')
            approximate_entropy = self._calc_phi(pattern_dimension, similarity_tolerance) - self._calc_phi(pattern_dimension + 1, similarity_tolerance)
            return  approximate_entropy

    def _calc_phi(self, pattern_dimension=2, similarity_tolerance=0.20):
        """
        给定数据，计算近似熵的中间量phi
        phi可以理解为在特定模式维数下，数据模式的多样性的衡量（越负越多样）
        Parameters  :
        ----------
        pattern_dimension: int
        模式维数
        similarity_tolerance: float
        相似容限阈值

        Returns  :
        -------
        phi: float
        """
        # 构造信号序列
        sig_list = np.asarray([self.values[i:i + pattern_dimension] for i in range(self.length - pattern_dimension + 1)])
        # 依次比较序列是否相似（即两个序列的对应端点的值相差均小于容差），统计相似数量，计算相似比例
        similar_ratio = []
        for seq in sig_list:
            diff_smaller_than_threshold = (sig_list - seq) <= (similarity_tolerance * self.SD)
            is_similar = diff_smaller_than_threshold.sum(axis=1) == pattern_dimension
            similar_ratio.append(is_similar.mean())
        # 计算phi
        similar_ratio = np.asarray(similar_ratio)
        phi = np.sum(np.log(similar_ratio)) / (self.length - pattern_dimension + 1)
        return phi

    # 经验模态分解
    def EMD(self):
        """
        经验模态分解(Empirical Mode Decomposition)

        Parameters  :
        ----------
        
        Returns  :
        -------
        (imfs, res): (ndarray, 1d-array)
        内涵模态分量(Intrinsic Mode Functions, IMF)的矩阵, 每一行代表一个分量
        
        To do: 把输出调整成wavesis的风格(输出成值、wav类或wav类的列表)
        """
        # 直接调用pyEMD库的算法
        pyemd = EMD()
        pyemd.emd(self.values)
        imfs, res = pyemd.get_imfs_and_residue()
        return imfs, res

    # 时频分析
    def time_frequency_analysis(self, NFFT=256, Fs=8000, mode='psd'):
        spectrum, freqs, times = plt.specgram(self.values, NFFT, Fs, noverlap=NFFT // 2, mode=mode)
        return freqs, times, spectrum

# To do: 测试代码
if __name__ == "__main__":
    pass