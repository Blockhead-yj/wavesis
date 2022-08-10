# -*- encoding: utf-8 -*-
'''
Projects -> File: -> threephasewav.py
Author: DYJ
Date: 2022/07/27 16:06:49
Desc: 三相电类
version: 1.0
'''

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .basewav import BaseWav
from . import frequencydomainwav as fdwav
from . import timedomainwav as fdwav

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
