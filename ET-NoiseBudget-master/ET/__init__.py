import os
import numpy as np
from gwinc.ifo.noises import *
from gwinc import load_budget

DEFAULT_FREQ = '1:3000:6000'

def get_local_filename(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)

class BudgetWrapper(nb.Noise):
    ifo_name = ''

    def load(self):
        self.budget = load_budget(self.ifo_name, freq=self.freq).run()
    
    def calc(self):
        return self.budget.psd

class ETLF(BudgetWrapper):
    """ET Low-Frequency"""
    ifo_name = 'ETLF' 

class ETHF(BudgetWrapper):
    """ET High-Frequency"""
    ifo_name = 'ETHF' 

class ETDesignReport(nb.Noise):
    style = dict(
        label='ET-D Design Report',
        color='black',
        linestyle='--',
        lw=1
    )

    def load(self):
        data = get_local_filename('et_d.txt')
        freq, asd = np.loadtxt(data).T
        self.psd = self.interpolate(freq, asd**2)

    def calc(self):
        return self.psd

def invsum(data):
    return 1.0/np.nansum([1.0/x for x in data], axis=0)

class ET(nb.Budget):
    name = 'Einstein Telescope'
    freq = DEFAULT_FREQ

    noises = [
        ETLF,
        ETHF
    ]

    # no to-strain conversion, as budgets are already in strain
    calibrations = [] 

    references = [
        ETDesignReport
    ]
    
    accumulate = invsum # calculate envelope of noise curves instead of sum
