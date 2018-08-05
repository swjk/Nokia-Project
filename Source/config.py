import os

#Hourly Data Location
hourlyDataLocation = os.path.join(os.getcwd(), "..", "Data", "HourlyData.xlsx")
hourlySheetName    = "reportRaw1"
#hourlyDataLocation = os.path.join(os.getcwd(), "..", "Data", "TestData.xlsx")
#hourlySheetName    = "Sheet1"

#Weekly Data Location
dailyDataLocation = os.path.join(os.getcwd(), "..", "Data", "DailyData.xlsx")
dailySheetName    = "Data"

#Write Stats About Algorithm Location
writeStatsDataLocation_Ref        = os.path.join(os.getcwd(), "..", "Data", "Results-Ref.xlsx")
writeStatsDataLocation_S_LSTM     = os.path.join(os.getcwd(), "..", "Data", "Results-S_LSTM.xlsx")
writeStatsDataLocation_S_ARIMA    = os.path.join(os.getcwd(), "..", "Data", "Results-S_ARIMA.xlsx")
writeStatsDataLocation_LR_LSTM    = os.path.join(os.getcwd(), "..", "Data", "Results-LR_LSTM.xlsx")
writeStatsDataLocation_LR_ARIMA   = os.path.join(os.getcwd(), "..", "Data", "Results-LR_ARIMA.xlsx")
writeStatsDataSheetName    = "Output"

#Write Correlation Data
writeCorrelationDailyLocation  = os.path.join(os.getcwd(), "..", "Data", "CorrelationDaily.xlsx")
writeCorrelationSDailySheetName = "Sheet1"

writeCorrelationHourlyLocation = os.path.join(os.getcwd(), "..", "Data", "CorrelationHourly.xlsx")
writeCorrelationHourlySheetName = "Sheet1"

#General
predictionHours = 24
trainingDays = 14

#Reference Algorithm
referenceWindowSize = 4
referenceGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "ReferenceResults")

#Linear Regression Algorithm
topNCorrelations = 3
lrARIMAGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "LinearRegressionARIMAResults")
lrLSTMGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "LinearRegressionLSTMResults")

#LSTM Algorithm
lstmGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "StraightLSTMResults")

#ARIMA Algorithm
arimaGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "StraightARIMAResults")

#Final Result
finalResultGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults")

#Correlation Threshold
dailyCorrThreshold  = 0.0
hourlyCorrThreshold = 0.0


#ForecastLimits
#Multiplication Constant with std
forcastLimitMultiplicationConstant = 3
#WindowSize of Rolling std
forcastLimitWindowSize = 6
