# -*- coding: utf-8 -*-
"""
@author: vchan
"""

import pandas

risk_df = pandas.read_csv("data_raw.csv")

#Subset the required columns
X = risk_df.iloc[:,2:5]
y = risk_df['risk']

#Save output
X.to_csv("X.csv")
y.to_csv("y.csv")
