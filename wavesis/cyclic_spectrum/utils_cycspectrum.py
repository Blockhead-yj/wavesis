from concurrent.futures import ThreadPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from tqdm import trange, tqdm

from pycycspectrum import CPS_W_parrallel, CPS_W_cupy, SCoh_W

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CyclicSpectrumCalculator(object):
    def __init__(self, fs=8000, Nw=8000, Nv=None, nfft=None, use_thread_pool=False, n_worker=4) -> None:
        '''
        fs: int
            采样频率
        Nw: int
            窗口长度
        Nv: int/None
            窗口重叠长度
        nfft: int/None
            傅里叶变换长度
        '''
        self.fs = fs
        self.Nw = Nw
        self.Nv = Nv if Nv is not None else np.floor(2 / 3 * Nw)
        self.nfft = nfft if nfft is not None else 2 * Nw
        if use_thread_pool:
            self.thread_pool = ThreadPoolExecutor(max_workers=n_worker)
        else:
            self.thread_pool = None
            

    def calc_CPS(self, data, cyc_freq_range=[0, 200], normalize=True, parrallel=False, n_worker=4):
        '''
        return f, alpha, S
        '''
        data_length = len(data)
        da = 1 / data_length # 频率分辨率
        # 根据指定的循环频率范围和循环频率分辨率计算循环频率的取值点集
        alpha = np.arange(cyc_freq_range[0], cyc_freq_range[1], self.fs * da)
        if parrallel and (not self.thread_pool is None):
            partial_CPS_W = partial(CPS_W_parrallel, x=data, y=data, nfft=self.nfft, Noverlap=self.Nv, Nwindow=self.Nw, opt='sym')
            normalized_alpha = alpha/self.fs
            # 使用线程执行map计算
            results = self.thread_pool.map(partial_CPS_W, normalized_alpha)
            S = np.asarray([i_res['S'] for i_res in tqdm(results, total=len(normalized_alpha), desc='循环频率迭代计算中', leave=False)]).T
            f = (np.float64(self.fs) * np.arange(self.nfft) / self.nfft)[:np.int32(self.nfft)//2]
        else:
            # 循环频率内循环
            S = np.zeros((int(self.nfft), len(alpha)), dtype=np.complex128)
            for i in trange(len(alpha), desc='循环频率迭代计算中', leave=False):
                CPS = CPS_W_cupy(data, data, alpha[i]/self.fs, self.nfft, self.Nv, self.Nw, 'sym')
                S[:, i] = np.asarray(CPS['S']).squeeze()        
            f = (np.float64(self.fs) * CPS['f'])[:np.int32(self.nfft)//2]
        f = f.reshape(-1, 1)
        S = np.abs(S[:np.int32(self.nfft)//2, :]) 
        if normalize:
            S /= np.max(S)  # 归一化
        return f, alpha, S
    
    def calc_Coh(self, data, cyc_freq_range=[0, 200], normalize=True):
        '''
        return f, alpha, Coh, Syx
        '''
        data_length = len(data)
        da = 1 / data_length # 频率分辨率
        # 根据指定的循环频率范围和循环频率分辨率计算循环频率的取值点集
        alpha = np.arange(cyc_freq_range[0], cyc_freq_range[1], self.fs * da)
        # 循环频率内循环
        C = np.zeros((int(self.nfft), len(alpha)), dtype=np.complex128)
        S = np.zeros((int(self.nfft), len(alpha)), dtype=np.complex128)
        for i in trange(len(alpha), desc='循环频率迭代计算中', leave=False):
            Coh = SCoh_W(data, data, alpha[i]/self.fs, self.nfft, self.Nv, self.Nw, 'sym')
            C[:, i] = np.asarray(Coh['C']).squeeze()
            S[:, i] = np.asarray(Coh['Syx']).squeeze()        
        f = (np.float64(self.fs) * Coh['f'])[:, :np.int32(self.nfft)//2]
        f = f.reshape(-1, 1)
        S = np.abs(S[:np.int32(self.nfft)//2, :]) 
        if normalize:
            S /= np.max(S)  # 归一化
        return f, alpha, S, C
    
    @staticmethod
    def plot_3Dspectrum(f,alpha, CS, freq_range=[10, 500], title=''):
        freq_index = (f >= freq_range[0]) & (f <= freq_range[1])
        CS = CS[freq_index.squeeze(), :]
        f = f[freq_index.squeeze()]
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        surf = ax.plot_surface(f, alpha, 20 * np.log10(CS), cmap=cm.get_cmap('GnBu'),
                            linewidth=0, antialiased=False)
        # Customize the z axis.
        # ax.set_zlim(0, np.max(CS)*0.85)
        ax.set_xlabel('频率(Hz)', fontdict={'fontsize':15})
        ax.set_ylabel('循环频率(Hz)', fontdict={'fontsize':15})
        ax.set_title('循环谱-' + title + '(dB)', fontdict={'fontsize':20})
        ax.zaxis.set_major_locator(LinearLocator(10))
        # # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.0f}')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return None
    @staticmethod
    def plot_slice_by_alpha(f, alpha, CS, target_alpha, freq_range=[10, 500], smooth=0, *args, **kwargs):
        freq_index = (f >= freq_range[0]) & (f <= freq_range[1])
        assert smooth >= 0
        
        if smooth:
            alpha_index = (alpha >= target_alpha - smooth) & (alpha <= target_alpha + smooth)
            plot_dat = np.mean(CS[freq_index.squeeze(), alpha_index.squeeze()], axis=1, keepdims=False)
        else:
            assert target_alpha in alpha
            alpha_index = (alpha == target_alpha)
            plot_dat = CS[freq_index.squeeze(), alpha_index.squeeze()]
        plt.plot(f[freq_index.squeeze()].squeeze(), 20 * np.log10(plot_dat.squeeze()), *args, **kwargs)
        plt.xlabel('频率(Hz)', fontdict={'fontsize':15})
        plt.ylabel('幅值(dB)', fontdict={'fontsize':15})
        plt.title(f'循环频率-{target_alpha}Hz 切片图', fontdict={'fontsize':20})
        return None
    @staticmethod
    def plot_mean_by_alpha(f, alpha, CS, freq_range=[10, 50], *args, **kwargs):
        plot_dat = np.mean(CS, axis=1)
        freq_index = (f >= freq_range[0]) & (f <= freq_range[1])
        plot_dat = plot_dat[freq_index.squeeze()]
        plt.plot(f[freq_index.squeeze()].squeeze()[10:], 20 * np.log10(plot_dat.squeeze())[10:], *args, **kwargs)
        plt.xlabel('频率(Hz)', fontdict={'fontsize':15})
        plt.ylabel('幅值(dB)', fontdict={'fontsize':15})
        plt.title(f'循环频率谱-求和图', fontdict={'fontsize':20})
        return None
    @staticmethod
    def plot_mean_by_freq(f, alpha, CS, *args, **kwargs):
        plot_dat = np.mean(CS, axis=0)
        plt.plot(alpha.squeeze()[10:], 20 * np.log10(plot_dat.squeeze())[10:], *args, **kwargs)  # 循环频率为0无意义
        plt.xlabel('循环频率(Hz)', fontdict={'fontsize':15})
        plt.ylabel('幅值(dB)', fontdict={'fontsize':15})
        plt.title(f'频率谱-求和图', fontdict={'fontsize':20})
        return None    
    
    @staticmethod
    def plot_sum_by_freq(f, alpha, CS, *args, **kwargs):
        plot_dat = np.sum(CS, axis=0)
        plt.plot(alpha.squeeze()[10:], 20 * np.log10(plot_dat.squeeze())[10:], *args, **kwargs)  # 循环频率为0无意义
        plt.xlabel('循环频率(Hz)', fontdict={'fontsize':15})
        plt.ylabel('幅值(dB)', fontdict={'fontsize':15})
        plt.title(f'频率谱-求和图', fontdict={'fontsize':20})
        return None
    
