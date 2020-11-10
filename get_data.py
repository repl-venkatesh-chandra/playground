# -*- coding: utf-8 -*-
"""
@author: vchan
"""

import pandas


risk_df = pandas.read_csv("train.csv")

risk_df.to_csv("data_raw.csv")



