#############Layout################

1* Algorithm found  : Source/predictor.py

2* Config file found: Source/config.py
All other dependent files are also found in Source/*.py 
(can ignore NNSource and PlotSource folder)


3* Data found: Data/*

4* Visual and Experimental Results just for intuition found: DailyResults/*, HourlyResults/*




1* contains the main function which can be run : python3 predictor.py all or python3 predictor.py ref

1* contains currently:
	Reference Algorithm
	
			Single ARIMA forecast Algorithm
 (e.g can be used when no dependence)
			Single LSTM forecast Algorithm	(e.g can be used when no dependence)
			Combined ARIMA with Linear Regression Algorithm



			Combined LSTM with Linear Regression Algorithm

2* contains constants that can be altered including location to store plots : Default in AlgorithmResults folder




########Install Dependencies##############

pip install pandas matplotlib scikit-learn scipy statsmodels keras patsy tensorflow xlrd


########Current Limitations##############


Prediction Forecast Must BE Hours = 24 and data must be hourly.
This is because of LSTM implementation.
ARIMA will work for <= 24

Linear Regression Algorithm will only work on KPIs that have one or more correlating KPIs. 
One should use Straight ARIMA,Straight LSTM or reference to model rest
