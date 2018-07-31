import os

#Hourly Data Location
hourlyDataLocation = os.path.join(os.getcwd(), "..", "Data", "HourlyData.xlsx")
hourlySheetName    = "reportRaw1"

#Weekly Data Location
dailyDataLocation = os.path.join(os.getcwd(), "..", "Data", "DailyData.xlsx")
dailySheetName    = "Data"

#Write Stats About Algorithm Location
writeStatsDataLocation     = os.path.join(os.getcwd(), "..", "Data", "NewData.xlsx")
writeStatsDataSheetName    = "Sheet1"

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
topNCorrelations = 5
lrARIMAGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "LinearRegressionARIMAResults")
lrLSTMGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "LinearRegressionLSTMResults")

#LSTM Algorithm
lstmGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "StraightLSTMResults")

#ARIMA Algorithm
lrARIMAGraphLocation = os.path.join(os.getcwd(), "..", "AlgorithmResults", "StraightARIMAResults")

#Correlation Threshold
dailyCorrThreshold  = 0.9
hourlyCorrThreshold = 0.96


#ForecastLimits
#Multiplication Constant with std
forcastLimitMultiplicationConstant = 3
#WindowSize of Rolling std
forcastLimitWindowSize = 6
