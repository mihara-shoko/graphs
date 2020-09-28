#!/usr/bin/env python
# coding: utf-8

# In[26]:

# calcurate the midpoint potential of protein and draw the titration curves

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

df = pd.read_excel('file_path', sheet_name='sheet_name')
df

xm = np.array(df['Em'].values)
ym1 = np.array(df['#1'].values)
ym2 = np.array(df['#2'].values)
ym3 = np.array(df['#3'].values)
y_ave = np.mean(np.vstack([ym1, ym2, ym3]),axis=0)
y_sd = np.std(np.vstack([ym1, ym2, ym3]), axis=0)

def fit_func (x, Em):
    func = 1/(10**(0.0338*(x-Em))+1)
    return func
    
def residue (Em, y, x):
    res = (y - fit_func(x, Em))
    return res

def calcurate (x, y):
    calcurated_Em = scipy.optimize.leastsq(residue, -300, args=(y, x), full_output=True)
    Em = calcurated_Em[0]
    return Em

def plot_graph (x, y, Em, num,  sd):
    print('\n#', num)
    xseq = np.arange(x[0], x[-1], 1, dtype=np.int64)
    y_app = fit_func(xseq, Em)
    plt.plot(x, y,'ko')
    plt.plot(xseq, y_app)
    if sd is not None:
        plt.errorbar(x, y, yerr=sd, capthick=1, capsize=5, fmt='none', ecolor='k')
    plt.show()

Em1 = calcurate(xm, ym1)
plot_graph(xm, ym1, Em1, 1, None)

Em2 = calcurate(xm, ym2)
plot_graph(xm, ym2, Em2, 2, None)

Em3 = calcurate(xm, ym3)
plot_graph(xm, ym3, Em3, 3, None)

Em_ave = np.mean([Em1, Em2, Em3])
Em_sd = np.std([Em1, Em2, Em3])
plot_graph(xm, y_ave, Em_ave, 'ave', y_sd)


print('Em1 = ' , Em1)
print('Em2 = ' , Em2)
print('Em3 = ' , Em3)
print('Em_ave = ', Em_ave, ' ± ', Em_sd)




# In[ ]:




