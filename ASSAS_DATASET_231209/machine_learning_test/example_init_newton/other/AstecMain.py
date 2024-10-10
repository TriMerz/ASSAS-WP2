#!/usr/bin/env python3

import abc
import os
# Importing numpy generates a FPE in core/getlimits.py
# This FPE is catch by astec, because during initialisation phase, FPE trap is enabling by astec.
# cf: https://github.com/numpy/numpy/issues/18202#issuecomment-858289775
# To avoid the problem, numpy must be imported before activating FPE trap, so before astec initialization.
# This is necessary to use nuclea or matplotlib for example, because they use numpy
try:
    import numpy
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import *
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import RootMeanSquaredError
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import load_model

except ImportError:
    pass
import pyastec as pyas

class AstecMain(abc.ABC):

    def __init__(self, mdat=None):
        self.icont = -2
        self.mdat = mdat

    def run(self):
        pyas.configure_signals()
        pyas.enable_fpe()
        self._dataset_reading()
        self._steady()
        self._computation()
        self._finalization()

    def _dataset_reading(self):
        if self.mdat is None:
            pyas.init()
        elif os.path.isfile(self.mdat):
            pyas.init(self.mdat)
        else:
            raise FileNotFoundError(self.mdat)

    def _steady(self):
        self.icont = pyas.steady()

    @abc.abstractmethod
    def _computation(self):
        pass

    def _finalization(self):
        pyas.end()
        print("End of astec computation with " +  self.__class__.__name__)
