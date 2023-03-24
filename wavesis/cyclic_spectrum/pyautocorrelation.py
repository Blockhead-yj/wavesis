import cupy as cp
from cupyx.scipy import linalg
import numpy as np

def cyclic_autocorrelation(x, max_tau=8000, sample_frequency=8000):
    '''计算循环自相关函数和循环谱密度函数
    '''
    # 将x转化为cupy.array，使用gpu运算
    x = cp.array(x)
    shift_x = linalg.toeplitz(cp.roll(x[::-1], 1)[:max_tau], x)
    R = shift_x * x
    # 计算循环自相关函数(Cyclic Autocorrelation Function 简称CAF), 对t进行傅里叶变换
    CAF = cp.abs(cp.fft.fft(R) / len(R))
    # 循环谱密度函数(Cyclic Spectrum Density Function 简称 CSDF), 对tau进行傅里叶变换
    # 循环谱密度函数又常常被称为谱相关密度函数(Spectrum Correlation Density Function 简称 SCDF)
    CSDF = cp.abs(cp.fft.fft(CAF, axis=0) / len(CAF))
    # 循环频率轴
    alpha = np.fft.fftfreq(len(x), d=1/sample_frequency)
    # 时延轴
    tau = np.arange(max_tau) / sample_frequency
    # 谱频率轴
    freq = np.arange(max_tau) / (max_tau / sample_frequency)
    return tau, alpha, freq, CAF.get(), CSDF.get()

def get_optimal_tau_slice(alpha_list, tau_list, CSDF, fault_freq=89, search_range=2):
    """_summary_

    Parameters
    ----------
    alpha_list : 1D-array
        循环频率序列
    tau_list : 1D-array
        时延序列
    CSDF : 2D-array
        循环谱密度函数(Cyclic Spectrum Density Function 简称 CSDF)
    fault_freq : int, optional
        故障特征频率
    search_range : int, optional
        故障特征频率附近搜寻范围, by default 2

    Returns
    -------
    optimal_tau_slice, max_alpha, max_tau
        时延切片, 最大取值对应的循环频率, 最大取值对应的时延
    """    
    # 获取故障频率点附近最大的时延参数对应的循环频率切片
    error_point_left = np.round(alpha_list,0).tolist().index(fault_freq-search_range)
    error_point_right = np.round(alpha_list,0).tolist().index(fault_freq+search_range)
    
    # 获取最大幅值对应的alpha
    target_CSDF = CSDF[:, error_point_left:error_point_right]
    _position = np.argmax(target_CSDF)
    _position_x, _position_y = np.unravel_index(_position, target_CSDF.shape)
    max_alpha = alpha_list[error_point_left:error_point_right][_position_y]
    
    # 获取时延切片最大值对应的时延参数
    max_tau = tau_list[_position_x]
    optimal_tau_slice = CSDF[_position_x, :]
    return optimal_tau_slice, max_alpha, max_tau
    