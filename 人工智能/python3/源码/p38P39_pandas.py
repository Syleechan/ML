# !/usr/bin/env python
# coding:utf-8
# Author: Caojian

import pandas as pd
df = pd.read_csv("name.csv",header=0)
print("name[0]:\n", df["name"][0],"\n")
print("name:\n", df["name"],"\n")
print("df.values:\n", df.values,"\n")