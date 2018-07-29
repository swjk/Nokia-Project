#Hourly Data Location
hourlyDataLocation = "../Data/HourlyData.xlsx"
hourlySheetName    = "reportRaw1"

#Weekly Data Location
dailyDataLocation = "../Data/DailyData.xlsx"
dailySheetName    = "Data"

#Write Data Location
writeNewDataLocation     = "../Data/NewData.xlsx"
writeNewDataSheetName    = "Sheet1"

#Write Correlation Data
writeCorrelationDailyLocation  = "../Data/CorrelationDaily.xlsx"
writeCorrelationSDailySheetName = "Sheet1"

writeCorrelationHourlyLocation = "../Data/CorrelationHourly.xlsx"
writeCorrelationHourlySheetName = "Sheet1"


#Reference
referenceWindowSize = 4
referenceGraphLocation = "../Plot/Reference/"


#Training/Future Data Separation
trainingDays = 14

#Arima Forcast Hours
predictionHours = 24

#Correlation Threshold
dailyThreshold  = 0.9
hourlyThreshold = 0.96
