import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPyton.display import display

# read data and drop redundant column
data = pd.read_csv('final_dataset.csv')

# Preview Data
display(data.head())

#Full Time Result (H=Home Win, D=Draw, A=Away Win)
#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction

#Input - 12 other features (fouls, shots, goals, misses,corners, red card, yellow cards)
#Output - Full Time Result (H=Home Win, D=Draw, A=Away Win)

# Total number of matches(Training + Test Set)
n_matches = data.shape[0]

# Calculate number of features
n_features = data.shape[1] - 1

# Calculate number of matches won by home team
n_homewins = len(data[data.FTR == 'H'])

# Calculate win rate for home team.
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print results
print('Total number of matches:  {}'.format(n_matches))
print('Total number of features: {}'.format(n_features))
print('Total number of matches won by home team: {}'.format(n_homewins))
print('Win rate of home team:    {}'.format(win_rate))

# Cisualising distribution of data
from pandas.tools.plotting import scatter_matrix

#the scatter matrix is plotting each of the columns specified against each other column.
#You would have observed that the diagonal graph is defined as a histogram, which means that in the 
#section of the plot matrix where the variable is against itself, a histogram is plotted.

#Scatter plots show how much one variable is affected by another. 
#The relationship between two variables is called their correlation
#negative vs positive correlation

#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction

scatter_matrix(data[['HTGD','ATGD','HTP','ATP','DiffFormPts','DiffLP']], figsize=(10,10))

# Separate into feature set and target variable
# FTR = Full Time Rseult (H=Home Win, D=Draw, A=Away Win)
X_all = data.drop(['FTR'],1)
y_all = data['FTR']

# Standardising the data
from sklearn.preprocessing import scale

# Center to the mean and component wise scale to unit variance
cols = [['HTGD','ATGD','HTP','ATP','DiffLP']]
for col in cols:
	X_all[col] = scale(x_all[col])

# Last 3 wins for both sides
X_all.HM1 = C_all.HM1.astype('str')
X_all.HM2 = C_all.HM2.astype('str')
X_all.HM3 = C_all.HM3.astype('str')
X_all.AM1 = C_all.AM1.astype('str')
X_all.AM2 = C_all.AM2.astype('str')
X_all.AM3 = C_all.AM3.astype('str')

# We watn continuos vars that are integers for out input data, so lets remove any categorical vars
def preprocess_features(X):
	"""
		Preprocesses the football data and converts catagorical variables into dummy variables.
	"""
	# Initialize new output DataFrame
	output = pd.DataFrame(index = X.index)

	# Investigate new output DataFrame
	for col, col_data in X.iteritems():
		# If data type is categorical, convert to dummy variables
		if col_data.dtype == object:
			col_data = pd.get_dummies(col_data, prefix = col)

			# Collect the revised columns
			output - output.join(col_data)
	return output

X_all = preprocess_features(X_all)
print("Processed feature columns({} total features)\n{}".format(len(X_all.columns), list(X_all.columns))

# Show the feature information by printing the first five rows
print("\nFeature values:")
display(X_all.head())


























