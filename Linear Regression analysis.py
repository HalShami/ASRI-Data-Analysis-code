import numpy
import pandas
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

numpy.random.seed(2023)

data = pandas.read_csv('Linear Regression.csv')

xdata = data[['SimpDivIndValue', 'AvgHHSize', 'BlackPct', 'AsianPct', 'HispanicPct', 'NativeAmericanPct', 'MultipleRacePct', 'MinorityPct', 'WhitePct', 'ForeignBornPct', 'UnemployedPct', 'PopEstimate2020', '65+Pct', 'EdLessHSPCT', 'EdHSPct', 'SomeCollegePct', 'EdAssoDegPct', 'EdCollege+Pct']]
ydata = data['PovValue']

fold = KFold(n_splits = 5, shuffle = True)
i = 1
mses = []
for trainindex, testindex in fold.split(xdata):
    print(i)
    i +=1
    trainxdata = xdata.iloc[trainindex]
    trainydata = ydata.iloc[trainindex]
    testxdata = xdata.iloc[testindex]
    testydata = ydata.iloc[testindex]

    model = LinearRegression()
    model.fit(numpy.asarray(trainxdata, dtype = 'float32'), numpy.asarray(trainydata, dtype = 'float32'))
    preds = model.predict(testxdata)

    mses.append(mean_squared_error(testydata, preds))
    print(model.coef_)
    print(model.score(numpy.asarray(testxdata, dtype = 'float32'), numpy.asarray(testydata, dtype = 'float32')))
    print(model.score(numpy.asarray(trainxdata, dtype = 'float32'), numpy.asarray(trainydata, dtype = 'float32')))
'''
print(numpy.asarray(mses).mean())
print(model.score())
model.coef_
'''

#polynomial features




