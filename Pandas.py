import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import nan as NA
import json
import matplotlib.pyplot as plt



# Series
lyst = np.arange(5)
index = ['d', 'b', 'a', 'c', 'e']
obj = Series(lyst, index=index)
print(obj)
print(obj.index)
print(obj.values)
print(obj[obj > 3])
print(obj**2)

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj1 = Series(sdata)
print(obj1)
print(obj.isnull())

# DataFrame
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
columns = ['year', 'state', 'pop']
df = DataFrame(data, columns=columns)

df['GDP'] = 10
print(df)

del df['GDP']
print(df)

# reindex
index = ['a', 'b', 'c', 'd', 'e', 'f']
obj = obj.reindex(index=index, fill_value=0)

print(obj)
print(obj[2])

# drop entries from an axis
new_obj = obj.drop(['e', 'f'])
print(new_obj)

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])

new_data = data.drop('two', axis=1)
print(new_data)

# arithmetic and alignment
df1 = DataFrame(np.arange(9.).reshape((3, 3)),
                columns=list('bcd'),
                index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)),
                columns=list('bde'),
                index=['Utah', 'Ohio', 'Texas', 'Oregon'])

print(df1 + df2)

# function application and mapping
frame = DataFrame(np.random.randn(4, 3),
                  columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])

f = lambda x: x.max() - x.min()
frame_apply = frame.apply(f, axis=1)

print(frame_apply)

# sorting and ranking
frame_sort = frame.sort_index()
print(frame_sort)

frame_sort = frame.sort_values(by='b')
print(frame_sort)

# Summarizing and computing describtive statistics
df = DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
               index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])

print(df.sum(axis=0))
print(df.mean(axis=1))
print(df.cumsum(axis=0))

# correlation and covariance
df = DataFrame(np.random.randn(100, 4), columns=['AAPL', 'GOOG', 'IBM', 'MSFT'])
df_corr = df.corr()
df_cov = df.cov()

print(df_corr)
print(df_cov)


# Unique values, values counts and membership
data = DataFrame(np.random.randint(0, 5, (5, 3)), columns=list('abc'))
print(data)
result = data.apply(pd.value_counts).fillna(0)
print(result)


# handling missing data
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
data_null = string_data.isnull()
print(data_null)

data = DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
print(cleaned)

# filling in missing data
data = data.fillna(0)
print(data)

# hierarchical indexing
data = Series(np.random.randn(10),
              index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                     [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])

print(data.unstack())

# using a dataframe's columns
frame = DataFrame({'a': range(7),
                   'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})

print(frame)
df = frame.set_index(['c', 'd'])
print(df)
df = df.reset_index().sort_index(axis=1).sort_values(by='d')
print(df)

# Data loading, storage, and file formats
# Reading and writing data in text format
df = pd.read_csv('cancer.csv')
print(df[:5])

# Writing Data out to text format
df.to_csv('dat.csv')

# JSON Data
obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""

result = json.loads(obj)
print(result)

# converts a python object back to JSON
ajson = json.dumps(result)

df = DataFrame(result['siblings'])
print(df)

# data Wrangling
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'],
                 'data2': range(3)})

df = pd.merge(df1, df2, on='key', how='left')
print(df)

df = df1.set_index(['key']).join(df2.set_index(['key']), how='outer')
print(df)

arr = np.arange(12).reshape((3, 4))
arr_con = np.concatenate([arr, arr], axis=1)

print(arr_con)

data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
data = data.drop_duplicates()
print(data)


data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham', 'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

meat_to_animal = {'bacon': 'pig',
                  'pulled pork': 'pig',
                  'pastrami': 'cow',
                  'corned beef': 'cow',
                  'honey ham': 'pig',
                  'nova lox': 'salmon'}

data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
print(data)

# Discretization and binning
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]

cats = pd.cut(ages, bins)
print(cats)

agg = pd.value_counts(cats)
print(agg)

# detecting and filtering outliers
np.random.seed(12345)
data = DataFrame(np.random.randn(100, 4))

dat = data[(np.abs(data) > 3).any(axis=1)]
print(dat)

# permutation and Random sampling
df = DataFrame(np.arange(20).reshape(5, 4))
sampler = np.random.permutation(5)
df = df.take(sampler)

print(df)

# Computing Indicator
df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': range(6)})
indicator = pd.get_dummies(df['key']).join(df['data1'])
print(indicator)

# string object methods
string = 'string is good. .....We use it to do lot of things.'
s = string.replace('.', '').split()
print(s)

# Visualization
# Figures
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
plt.plot(np.random.randn(50), 'k--')
ax1.hist(np.random.randn(50), bins=10, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()

fig, axes = plt.subplots(2, 3)
plt.show()

# Data Aggregation
df = pd.read_csv('heart.csv')

grouped = df.groupby(['sex', 'target'])
grouped_age = grouped['age'].agg(['count'])
print(grouped_age)

grouped_age.plot(kind='bar', rot=30)
plt.show()
