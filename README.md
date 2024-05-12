Q1.Write programs in Python using NumPy library to do the following:
a. Compute the mean, standard deviation, and variance of a two dimensional random integer array along 
the second axis.
b. Create a 2-dimensional array of size m x n integer elements, also print the shape, type and data type of 
the array and then reshape it into an n x m array, where n and m are user inputs given at the run time.
c. Test whether the elements of a given 1D array are zero, non-zero and NaN. Record the indices of these 
elements in three separate arrays.
d. Create three random arrays of the same size: Array1, Array2 and Array3. Subtract Array 2 from Array3 
and store in Array4. Create another array Array5 having two times the values in Array1. Find Covariance and Correlation of Array1 with Array4 and Array5 respectively.
e. Create two random arrays of the same size 10: Array1, and Array2. Find the sum of the first half of both 
the arrays and product of the second half of both the arrays.


import numpy as np

# Task a: Compute mean, standard deviation, and variance of a 2D random integer array along the second axis
arr_a = np.random.randint(0, 10, size=(3, 4))
mean_a = np.mean(arr_a, axis=1)
std_dev_a = np.std(arr_a, axis=1)
variance_a = np.var(arr_a, axis=1)
print("Task a:")
print("Mean along the second axis:", mean_a)
print("Standard Deviation along the second axis:", std_dev_a)
print("Variance along the second axis:", variance_a)

# Task b: Create, print shape, type, and data type of a 2D array, then reshape it based on user input
m = int(input("Enter number of rows (m): "))
n = int(input("Enter number of columns (n): "))
arr_b = np.random.randint(0, 10, size=(m, n))
print("\nTask b:")
print("Original Array:")
print("Shape:", arr_b.shape)
print("Type:", type(arr_b))
print("Data Type:", arr_b.dtype)
arr_b_reshaped = arr_b.reshape((n, m))
print("\nReshaped Array:")
print(arr_b_reshaped)


# Task c: Test whether elements of a 1D array are zero, non-zero, or NaN, and record indices
arr_c = np.array([0, 1, 2, 0, np.nan, 4, 0])
zero_indices = np.where(arr_c == 0)[0]
non_zero_indices = np.where(arr_c != 0)[0]
nan_indices = np.where(np.isnan(arr_c))[0]
print("\nTask c:")
print("Indices of zero elements:", zero_indices)
print("Indices of non-zero elements:", non_zero_indices)
print("Indices of NaN elements:", nan_indices)

# Task d: Create random arrays, perform subtraction, and find covariance and correlation
Array1_d = np.random.rand(5)
Array2_d = np.random.rand(5)
Array3_d = np.random.rand(5)
Array4_d = Array3_d - Array2_d
Array5_d = 2 * Array1_d
covariance_d = np.cov(Array1_d, Array4_d)[0, 1]
correlation_d = np.corrcoef(Array1_d, Array5_d)[0, 1]
print("\nTask d:")
print("Covariance of Array1 and Array4:", covariance_d)
print("Correlation of Array1 and Array5:", correlation_d)

# Task e: Create random arrays, find sum of first half and product of second half
Array1_e = np.random.rand(10)
Array2_e = np.random.rand(10)
sum_first_half_e = np.sum(Array1_e[:5]) + np.sum(Array2_e[:5])
product_second_half_e = np.prod(Array1_e[5:]) * np.prod(Array2_e[5:])
print("\nTask e:")
print("Sum of first half of Array1 and Array2:", sum_first_half_e)
print("Product of second half of Array1 and Array2:", product_second_half_e)

Q2.Do the following using PANDAS Series:
a. Create a series with 5 elements. Display the series sorted on index and also sorted on values seperately
b. Create a series with N elements with some duplicate values. Find the minimum and maximum ranks 
assigned to the values using ‘first’ and ‘max’ methods
c. Display the index value of the minimum and maximum element of a Series

import pandas as pd

# a. Create a series with 5 elements. Display the series sorted on index and also sorted on values separately
series_a = pd.Series([3, 1, 5, 2, 4], index=['b', 'a', 'e', 'c', 'd'])

# Sorted by index
sorted_by_index = series_a.sort_index()
print("Sorted by Index:")
print(sorted_by_index)

# Sorted by values
sorted_by_values = series_a.sort_values()
print("\nSorted by Values:")
print(sorted_by_values)

# b. Create a series with N elements with some duplicate values. Find the minimum and maximum ranks assigned to the values using ‘first’ and ‘max’ methods
series_b = pd.Series([3, 1, 5, 2, 4, 2, 1])  # Series with duplicate values
min_rank_first = series_b.rank(method='first').min()
max_rank_max = series_b.rank(method='max').max()
print("\nMinimum rank (using 'first' method):", min_rank_first)
print("Maximum rank (using 'max' method):", max_rank_max)

# c. Display the index value of the minimum and maximum element of a Series
min_index = series_a.idxmin()
max_index = series_a.idxmax()
print("\nIndex value of minimum element:", min_index)
print("Index value of maximum element:", max_index)

Q3. Create a data frame having at least 3 columns and 50 rows to store numeric data generated using a random 
function. Replace 10% of the values by null values whose index positions are generated using random function. 
Do the following:
a. Identify and count missing values in a data frame.
b. Drop the column having more than 5 null values.
c. Identify the row label having maximum of the sum of all values in a row and drop that row.
d. Sort the data frame on the basis of the first column.
e. Remove all duplicates from the first column.
f. Find the correlation between first and second column and covariance between second and third 
column. 
g. Discretize the second column and create 5 bins


import pandas as pd
import numpy as np

# Generate random numeric data
np.random.seed(0)
data = np.random.rand(50, 3)

# Create a DataFrame
df = pd.DataFrame(data, columns=['Column1', 'Column2', 'Column3'])

# Generate random indices for null values
null_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)

# Replace values at null indices with NaN
df.loc[null_indices] = np.nan

# a. Identify and count missing values in the DataFrame
missing_values_count = df.isnull().sum()
print("Missing values count:\n", missing_values_count)

# b. Drop the column having more than 5 null values
df = df.dropna(thresh=len(df) - 5, axis=1)

# c. Identify the row label having the maximum sum of all values in a row and drop that row
max_sum_row_label = df.sum(axis=1).idxmax()
df = df.drop(max_sum_row_label)

# d. Sort the DataFrame based on the first column
df = df.sort_values(by='Column1')

# e. Remove all duplicates from the first column
df = df.drop_duplicates(subset='Column1')

# f. Find the correlation between the first and second column
correlation = df['Column1'].corr(df['Column2'])
print("Correlation between the first and second column:", correlation)

# Find the covariance between the second and third column
covariance = df['Column2'].cov(df['Column3'])
print("Covariance between the second and third column:", covariance)

# g. Discretize the second column and create 5 bins
df['Column2_bins'] = pd.qcut(df['Column2'], q=5) 
print("FINAL DATAFRAME AFTER:\n", df)


Q4.Consider two excel files having attendance of two workshos. Each file has three fields ‘Name’, ‘Date, duration 
(in minutes) where names are unique within a file. Note that duration may take one of three values (30, 40, 50) 
only. Import the data into two data frames and do the following:
a. Perform merging of the two data frames to find the names of students who had attended both 
workshops.
b. Find names of all students who have attended a single workshop only.
c. Merge two data frames row-wise and find the total number of records in the data frame.
d. Merge two data frames row-wise and use two columns viz. names and dates as multi-row indexes. 
Generate descriptive statistics for this hierarchical data frame

import pandas as pd

# File paths and names
file1_path = r'C:\Users\Dell\Desktop\Python\main1.py\workshop1_attendance.xlsx'
file2_path = r'C:\Users\Dell\Desktop\Python\main1.py\workshop2_attendance.xlsx'

# Import data from Excel files into data frames
df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

# a. Perform merging of the two data frames to find the names of students who had attended both workshops
both_workshops = pd.merge(df1, df2, on='Name')
print("Names of students who attended both workshops:\n", both_workshops['Name'])

# b. Find names of all students who have attended a single workshop only
single_workshop = pd.concat([df1, df2]).drop_duplicates(keep=False)
print("Names of students who attended a single workshop only:\n", single_workshop['Name'])

# c. Merge two data frames row-wise and find the total number of records in the data frame
merged_df = pd.concat([df1, df2])
total_records = len(merged_df)
print("Total number of records in the merged data frame:", total_records)

# d. Merge two data frames row-wise and use two columns viz. names and dates as multi-row indexes.
# Generate descriptive statistics for this hierarchical data frame
merged_df = pd.concat([df1.set_index(['Name', 'Date']), df2.set_index(['Name', 'Date'])])
descriptive_stats = merged_df.describe()
print("Descriptive statistics for the hierarchical data frame:\n", descriptive_stats)

Q5.Using Iris data, plot the following with proper legend and axis labels: (Download IRIS data from: 
https://archive.ics.uci.edu/ml/datasets/iris or import it from sklearn datasets)
a. Plot bar chart to show the frequency of each class label in the data.
b. Draw a scatter plot for Petal width vs sepal width and fit a regression line
c. Plot density distribution for feature petal length.
d. Use a pair plot to show pairwise bivariate distribution in the Iris Dataset.
e. Draw heatmap for the four numeric attributes
f. Compute mean, mode, median, standard deviation, confidence interval and standard error for each 
feature
g. Compute correlation coefficients between each pair of features and plot heatmap


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

# Download Iris dataset from sklearn
iris = load_iris()

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# a. Plot bar chart to show the frequency of each class label in the data
class_counts = df['target'].value_counts()
class_labels = iris.target_names
plt.bar(class_labels, class_counts)
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class Label')
plt.show()

# b. Draw a scatter plot for Petal width vs sepal width and fit a regression line
sns.regplot(x='sepal width (cm)', y='petal width (cm)', data=df)
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Scatter Plot: Petal Width vs Sepal Width')
plt.show()

# c. Plot density distribution for feature petal length
sns.kdeplot(data=df['petal length (cm)'], shade=True)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Density')
plt.title('Density Distribution: Petal Length')
plt.show()

# d. Use a pair plot to show pairwise bivariate distribution in the Iris Dataset
sns.pairplot(df, hue='target')
plt.show()

# e. Draw heatmap for the four numeric attributes
numeric_attributes = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
correlation_matrix = df[numeric_attributes].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap: Correlation Matrix')
plt.show()

# f. Compute mean, mode, median, standard deviation, confidence interval, and standard error for each feature
feature_stats = df.describe().loc[['mean', '50%', 'std']]
mode = df.mode().iloc[0]
confidence_interval = stats.t.interval(0.95, len(df)-1, loc=df.mean(), scale=stats.sem(df))
standard_error = stats.sem(df)
feature_stats = feature_stats.append(pd.Series(mode, name='mode'))
feature_stats = feature_stats.append(pd.Series(confidence_interval[0], name='confidence_interval_lower'))
feature_stats = feature_stats.append(pd.Series(confidence_interval[1], name='confidence_interval_upper'))
feature_stats = feature_stats.append(pd.Series(standard_error, name='standard_error'))
feature_stats = feature_stats.rename(index={'50%': 'median'})
print("Descriptive statistics for each feature:\n", feature_stats)

# g. Compute correlation coefficients between each pair of features and plot heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap: Correlation Coefficients')
plt.show()



Q6.Consider the following data frame containing a family name, gender of the family member and her/his monthly 
income in each record.
Name Gender MonthlyIncome (Rs.)
Shah Male 114000.00
Vats Male 65000.00
Vats Female 43150.00
Kumar Female 69500.00
Vats Female 155000.00
Kumar Male 103000.00
Shah Male 55000.00
Shah Female 112400.00
Kumar Female 81030.00
Vats Male 71900.00
Write a program in Python using Pandas to perform the following:
a. Calculate and display familywise gross monthly income.
b. Calculate and display the member with the highest monthly income.
c. Calculate and display monthly income of all members with income greater than Rs. 60000.00.
d. Calculate and display the average monthly income of the female members


import pandas as pd

# Create the DataFrame
data = {
    'Name': ['Shah', 'Vats', 'Vats', 'Kumar', 'Vats', 'Kumar', 'Shah', 'Shah', 'Kumar', 'Vats'],
    'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'MonthlyIncome (Rs.)': [114000.00, 65000.00, 43150.00, 69500.00, 155000.00, 103000.00, 55000.00, 112400.00, 81030.00, 71900.00]
}

df = pd.DataFrame(data)

# a. Calculate and display familywise gross monthly income
familywise_income = df.groupby('Name')['MonthlyIncome (Rs.)'].sum()
print("Familywise gross monthly income:\n", familywise_income)

# b. Calculate and display the member with the highest monthly income
highest_income_member = df.loc[df['MonthlyIncome (Rs.)'].idxmax()]
print("Member with the highest monthly income:\n", highest_income_member)

# c. Calculate and display monthly income of all members with income greater than Rs. 60000.00
high_income_members = df[df['MonthlyIncome (Rs.)'] > 60000.00]
print("Monthly income of members with income greater than Rs. 60000.00:\n", high_income_members)

# d. Calculate and display the average monthly income of the female members
average_female_income = df[df['Gender'] == 'Female']['MonthlyIncome (Rs.)'].mean()
print("Average monthly income of female members:", average_female_income)


Q7. Using Titanic dataset, to do the following:
a. Find total number of passengers with age less than 30
b. Find total fare paid by passengers of first class
c. Compare number of survivors of each passenger class
d. Compute descriptive statistics for any numeric attribute genderwise

import pandas as pd

# File path and name
file_path = r'C:\Users\Dell\Desktop\Python\main1.py\titanic.csv'

# Import Titanic dataset
df = pd.read_csv(file_path)

# a. Find total number of passengers with age less than 30
passengers_less_than_30 = df[df['Age'] < 30]
total_passengers_less_than_30 = len(passengers_less_than_30)
print("Total number of passengers with age less than 30:", total_passengers_less_than_30)

# b. Find total fare paid by passengers of first class
first_class_fare = df[df['Pclass'] == 1]['Fare'].sum()
print("Total fare paid by passengers of first class:", first_class_fare)

# c. Compare number of survivors of each passenger class
survivors_by_class = df.groupby('Pclass')['Survived'].sum()
print("Number of survivors by passenger class:\n", survivors_by_class)

# d. Compute descriptive statistics for any numeric attribute genderwise
numeric_attribute = 'Age'
descriptive_stats_genderwise = df.groupby('Sex')[numeric_attribute].describe()
print("Descriptive statistics for", numeric_attribute, "genderwise:\n", descriptive_stats_genderwise)
