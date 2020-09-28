#!/usr/bin/env python
# coding: utf-8

# calculate the midpoint potential and draw the titration curves


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

def read_excel_file (file_path, sheet):
    df = pd.read_excel(file_path, sheet_name=sheet)
    return df

def cal_plot (df, x_name, y1_name, y2_name, y3_name, Em):
    xm = np.array(df[x_name].values)
    ym1 = np.array(df[y1_name].values)
    ym2 = np.array(df[y2_name].values)
    ym3 = np.array(df[y3_name].values)
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
        fig = plt.figure(figsize=(4.0, 4.0), dpi=300)
        ax = fig.add_subplot(111, xlabel='Em (mV)', ylabel='Reduction level')
        ax.tick_params(direction='in')
        ax.plot(x, y,'ko')
        ax.plot(xseq, y_app, 'k')
        if sd is not None:
            ax.errorbar(x, y, yerr=sd, capthick=1, capsize=5, fmt='none', ecolor='k')
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








