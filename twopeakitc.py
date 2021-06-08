import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
import pandas as pd

#data - Pandas dataframe imported from NanoAnalyze
#const - dict of constants
#**kwargs - values, mins, maxes
def fullFit(data, const, nFits = 1, chiMax = np.inf, **kwargs):
    dV = const['dV']    #Injection Volume
    V0 = const['V0']    #Cell Volume
    c = const['c']  #Syringe Concentration
    mol = dV * c   #Moles per injection

    Xt = data['Moles (Syringe)'] / V0     #Total ligand conc.
    Mt = data['Moles (Cell)'] / V0        #Total macromolecule conc.

    yvals = np.asarray(data['Y: Area Data (µJ)'])[1:] * 1e-6
    ydata = yvals / mol
    
    def NDH(Xt, N, K, dH):
        #Q = N M_t dH V0 / 2 [1 + X_t / N M_t + 1 / K N M_t - sqrt((1 + X_t / N M_t + 1 / K N M_t)^2 - 4 X_t / N M_t)]
        term1 = (N * Mt * dH * V0) / 2
        term2 = Xt / (N * Mt)
        term3 = 1 / (K * N * Mt)
        Q = term1 *(1 + term2 + term3 - ((1 + term2 + term3)**2 - 4 * term2)**0.5)
        #Normalized Heat - dQ / (moles in ith injected volume)
        #dQ is corrected for change in volume by adding the term dV/V0((Q(i)-Q(i-1)))/2
        q = np.roll(Q, 1) #array representing Q[i-1]
        dq = Q + (dV/V0) * ((Q + q) / 2) - q
        ndh = dq / (c * dV)
        #note that we do not return the first point -- it is not valid (Q[-1] does not exist)
        return ndh[1:]

    def NDHTotal(Xt, N1, K1, dH1, N2p, K2, dH2, N3):
        NDH1 = NDH(Xt, N1, K1, dH1) #Fits the first peak
        NDH3 = NDH(Xt, N3, K1, dH1)
        NDH2p= NDH(Xt, N2p,K2, dH2)

        curve3 = abs((dH1 - NDH3) / dH1) #curve defined by abs((dH1 - NDH3)/dH1)

        NDH2 = curve3 * NDH2p #multiply NDH2` by curve3 (fits the second peak)

        return NDH1 + NDH2 #Sum of NDH1 and NDH2 (fits the entire data)

    def toParams(**kwargs):
        params = lm.Parameters()
        params.add('N1', min=1e-300)
        params.add('K1', min=1e-300)
        params.add('dH1', min=1e-300)
        params.add('delta', min=1e-300)
        params.add('N2p', min=1e-300, expr='N1 + delta')
        params.add('K2', min=1e-300)
        params.add('dH2', min=1e-300)
        params.add('N3', min=1e-300)

        try:
            values = kwargs['values']
            for key in values:
                try:
                    params[key].set(value=values[key])
                except KeyError:
                    pass
        except KeyError:
            pass
        try:
            mins = kwargs['mins']
            for key in mins:
                try:
                    params[key].set(min=mins[key])
                except KeyError:
                    pass
        except KeyError:
            pass
        try:
            maxes = kwargs['maxes']
            for key in maxes:
                try:
                    params[key].set(max=maxes[key])
                except KeyError:
                    pass
        except KeyError:
            pass

        return params
    
    def iterativeFit(model, ydata, params, Xt, nFits, chiMax):
        first_fit = model.fit(ydata, params, Xt=Xt)
        best_fit = first_fit
        prev_fit = first_fit
        n = 1
        while n < nFits:
            new_fit = model.fit(ydata, prev_fit.params, Xt=Xt)
            
            if new_fit.redchi < best_fit.redchi:
                best_fit = new_fit
                print('New fit found')
            
            prev_fit = new_fit
            n += 1
            
            print('Fit %i completed' %n)
        
        if best_fit.redchi > chiMax:
            print('Fit Warning: Reduced Chi-Square not accepted')
        
        return first_fit, best_fit
        
    
    model = lm.Model(NDHTotal)
    params = toParams(**kwargs)
    
    result = iterativeFit(model, ydata, params, Xt, nFits, chiMax)
    
    return result

#data - pandas dataframe
#result - lmfit.ModelResult
def fitOut(data, result, verbose=True, dataPlot=True, bestPlot=True, firstPlot=False):
    ydata = result.data
    if verbose == True:
        print(result.fit_report())
    
    if dataPlot == True:
        plt.plot(data['Mole Ratio'][1:], ydata, 'ko', label='Area Data')
    
    if bestPlot == True:
        plt.plot(data['Mole Ratio'][1:], result.best_fit, 'b-', label='Best Fit')
        
    if firstPlot == True:
        plt.plot(data['Mole Ratio'][1:], result.first_fit, 'r--', label='Best Fit')
    
    if dataPlot == True or bestPlot == True or firstPlot == True:
        plt.xlabel('Mole Ratio')
        plt.ylabel('µJ / mol')
        plt.legend(loc='best')

#result - lmfit.ModelResult
#filename - file to save
#sep - delimiter character
def saveFit(result, filename, sep = ','):
    data = result.data
    bestOut = result.best_fit
    firstOut = result.init_fit

    arrayOut = np.array((data, bestOut, firstOut))
    arrayOut = np.transpose(arrayOut)
    arrayOut = pd.DataFrame(arrayOut, columns=('Data', 'Best Fit', 'First Fit'))

    out = open(filename, 'w')
    out.write(pd.DataFrame.to_csv(arrayOut, sep=sep))
    out.close()