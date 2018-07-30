#Hourly Data Location
hourlyDataLocation = "../Data/HourlyData.xlsx"
hourlySheetName    = "reportRaw1"

#Weekly Data Location
dailyDataLocation = "../Data/DailyData.xlsx"
dailySheetName    = "Data"

#Write Stats About Algorithm Location
writeStatsDataLocation     = "../Data/NewData.xlsx"
writeStatsDataSheetName    = "Sheet1"

#Write Correlation Data
writeCorrelationDailyLocation  = "../Data/CorrelationDaily.xlsx"
writeCorrelationSDailySheetName = "Sheet1"

writeCorrelationHourlyLocation = "../Data/CorrelationHourly.xlsx"
writeCorrelationHourlySheetName = "Sheet1"

#General
predictionHours = 24
trainingDays = 14


#Reference Algorithm
referenceWindowSize = 4
referenceGraphLocation = "../AlgorithmResults/ReferenceResults/"

#Linear Regression Algorithm
topNCorrelations = 5
lrGraphLocation = "../AlgorithmResults/LinearRegressionARIMAResults/"

#Correlation Threshold
dailyCorrThreshold  = 0.9
hourlyCorrThreshold = 0.96


#ForecastLimits
#Multiplication Constant with std
forcastLimitMultiplicationConstant = 3
#WindowSize of Rolling std
forcastLimitWindowSize = 6
