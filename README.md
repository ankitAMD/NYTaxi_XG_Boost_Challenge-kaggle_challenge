# Nyc-taxi-Kaggle-challenge

# Goal

Kaggle competition to predict NYC taxi travel times. The report for the project is at capstone.pdf.

## PROBLEM S TATEMENT
In this report, we look at a Kaggle competition with data from the NYC Taxi and
Limousine Commission, which asks competitors to predict the total ride time
(trip_duration) of taxi trips in New York City. The data provided by Kaggle is structured
data provided as a CSV file. The data in the CSV file includes multiple formats: timestamps,
text, and numerical data. This is a regression analysis, since the output, total ride time, is
numerical. I will use several machine learning methods for the prediction task, which are
linear regression, k-nearest neighbors regression, random forests, and XGBoost. The
models will be evaluated using the root mean squared logarithmic error.

## Overview

i perform this project on dekstop with using Jupyter_Notebook and also perform  without using Jupyter_notebook  on remote server using python.

## Software and Libraries
- Python 3
- Scikit-learn: Python’s open source machine learning library
- XGBoost: Python package for XGBoost model,

## Datasets
The primary train dataset (train.csv) and test dataset (test.csv) is at the <a href="https://www.kaggle.com/c/nyc-taxi-trip-duration/data">Kaggle competition website</a>.

The weather dataset is at: <a href="https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016">weather_data_nyc_centralpark_2016.csv</a>.

The datasets for the fastest routes from OSRM can be found <a href=https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm>here</a>. The files are:  fastest_routes_train_part_1.csv, fastest_routes_train_part_2.csv, and fastest_routes_test.csv

## Visualization Image
The final visualization image for the project report is visualization.pdf (.png), and is best viewed zoomed in.
h

## Credits

Credits for this code go to [rebeccak1](https://github.com/rebeccak1/nyc-taxi) 

 Credits for this code go to:
 
[https://github.com/llSourcell/kaggle_challenge/blob/master/notebook.ipynb]
 
[https://www.youtube.com/watch?v=suRd3UzdBeo]

## Remote Server Configuration:
Learn copy  transfer file from host to remote server and vice versa [http://qr.ae/TUNL43].

## More Learning Resources:
https://www.kaggle.com/kanncaa1/machi... 

https://www.kaggle.com/rtatman/beginn... 

https://machinelearningmastery.com/ge...

http://blog.kaggle.com/2017/01/23/a-k... 

https://www.youtube.com/watch?v=suRd3UzdBeo....

This video is apart of my Machine Learning Journey course:

https://github.com/llSourcell/Machine...

## basic resources 

https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html .........(df.loc)

https://www.tutorialspoint.com/numpy/numpy_introduction.htm...

https://www.tutorialspoint.com/seaborn/seaborn_quick_guide.htm......

https://www.tutorialspoint.com/python_pandas/python_pandas_introduction.htm...

https://www.datacamp.com/community/tutorials/xgboost-in-python.... ## Necessarily read (Xgboost)

https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline/43028034 //Provided you are running IPython, the

%matplotlib inline will make your plot outputs appear and be stored within the notebook.

/Scikit-learn

https://www.datacamp.com/community/tutorials/machine-learning-python......

https://www.analyticsvidhya.com/blog/2015/01/scikit-learn-python-machine-learning-tool/....

scikit-learn is used to build models. It should not be used for reading the data, manipulating and summarizing it. There are 

better libraries for that (e.g. NumPy, Pandas etc.)

http://scikit-learn.org/stable/tutorial/basic/tutorial.html.....

https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/

https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/...

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html...

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html...

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html....

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html...




## remove a directory through terminal contain with many files

"rm -rf ankit"
