#!/usr/bin/env python
# coding: utf-8

# In[2]:

# draw a graph with cont. error bar (e.g. absorption spectrum of cyanobacterial cells)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('file_path')
print(df)
x = np.array(df['WL'])
x_rev = np.flip(x)
x_rev = np.append(x, x_rev)

#WT strain
y_WT = np.array(df['WTave'])
y_WTsd = np.array(df['WTsd'])
y_WTup = y_WT + y_WTsd
y_WTlow = np.flip(y_WT - y_WTsd)
y_WTuplow = np.append(y_WTup, y_WTlow)

#dx strain
y_dx = np.array(df['dxave'])
y_dxsd = np.array(df['dxsd'])
y_dxup = y_dx + y_dxsd
y_dxlow = np.flip(y_dx - y_dxsd)
y_dxuplow = np.append(y_dxup, y_dxlow)

plt.plot(x, y_WT, color='k')
plt.plot(x, y_dx, color='g')
plt.fill_between(x_rev, y_WTuplow, color='k', alpha=0.2)
plt.fill_between(x_rev, y_dxuplow, color='g', alpha=0.2)
plt.xlim(350,800)
plt.ylim(0.01, 1.2)
plt.show()


# In[ ]:




