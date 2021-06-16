ITCFit
Danielle Wallace
=======================================
Dependencies:
numpy - https://numpy.org/doc/stable/contents.html
lmfit - https://lmfit.github.io/lmfit-py/index.html
matplotlib - https://matplotlib.org/stable/contents.html
pandas - https://pandas.pydata.org/pandas-docs/stable/index.html
=======================================
Methods:

fullFit(data, const, **kwargs)
	Takes a set of data and performs a fitting routine on it.
	Parameters: data (pandas.DataFrame) - input data to be fit
	            const (dict) - constants required for the model function (dV, V0, c)
		    **kwargs (optional) - keyword arguments passed to toParams
	Returns: lmfit.ModelResult 

fitOut(data, result, verbose=True, dataPlot=True, bestPlot=True, firstPlot=False)
	Takes a set of data and its corresponding fit output and prints a fit report as well as plots the best fit.
	Parameters: data (pandas.DataFrame) - input data that has been fit
		    result (lmfit.ModelResult) - fit output from fullfit
		    verbose (bool, optional) - whether or not to print a fit report
		    dataPlot (bool, optional) - whether or not to plot the input data
		    bestPlot (bool, optional) - whether or not to plot the best fit attempt
		    firstPlot (bool, optional) - whether or not to plot the first fit attempt

saveFit(result, filename, sep = ',')
	Takes a fit result and saves it to a text file.
	Parameters: result (lmfit.ModelResult) - fit output to be saved
		    filename (str) - name to save the output as
		    sep (str, optional) - string of length 1. Field delimiter for the output file.
