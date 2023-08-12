#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from collections import  Counter
from sklearn.feature_extraction.text import CountVectorizer

url="https://raw.githubusercontent.com/lichengrui/glassdoor_scraping/main/data.csv"

df =pd.read_csv(url)

cols = df.columns.tolist()

df = df.drop(columns=cols[0])

cols = df.columns.tolist()

df = df.dropna(subset=cols, how='all')

display(df)


# In[2]:


column_data_types = df.dtypes

print(column_data_types)


# In[3]:


# The graph with kde in a seaborn version isn't very meaningful

# Create a histogram of the 'rating' column
plt.hist(df['rating'], bins=5, edgecolor='black') 

plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')

# Show the plot
plt.show()


# In[4]:


# Calculate the length (number of words) in each document
df['con_word_count'] = df['conwords'].apply(lambda x: len(x.split()))

print(df['con_word_count'].max())


# In[5]:


sns.set(style="whitegrid")

# Create a histogram of the 'word_count' column
plt.figure(figsize=(10, 6))  # Set the figure size
sns.histplot(data=df, x='con_word_count', bins=20, kde=True)  # Use Seaborn to create the histogram
plt.xlabel('Con Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Document Con Word Counts')
plt.show()



# Create a histogram of the word counts
plt.hist(df['con_word_count'], bins=20, edgecolor='black') 

plt.xlabel('Number of Con Words')
plt.ylabel('Frequency')
plt.title('Distribution of Document Con Word Counts')

plt.show()


# In[6]:


# I see that there are not many data points that fall into 1200 range, so cutting the length
short_word_df = df[df['con_word_count'] < 350]

# Create a histogram of the word counts
plt.hist(short_word_df ['con_word_count'], bins=20, edgecolor='black')  # You can adjust the number of bins as needed

# Add labels and title
plt.xlabel('Number of Con Words')
plt.ylabel('Frequency')
plt.title('Subset Distribution of Document Con Word Counts')

# Show the plot


# In[7]:


df['employLengthStatus'].value_counts()


# In[8]:


# Function to extract numeric values and phrases
def extract_numeric_and_phrase(text):
    numeric_value = re.search(r'\d+', text)  # Extract numeric values
    more_than = re.search(r'more than', text, re.IGNORECASE)  # Check for "more than"
    less_than = re.search(r'less than', text, re.IGNORECASE)  # Check for "less than"
    
    numeric = int(numeric_value.group()) if numeric_value else None
    
    if more_than:
        phrase = 'more than'
    elif less_than:
        phrase = 'less than'
    else:
        phrase = None
    
    return numeric, phrase

# Apply the function to the 'term' column and create new columns
df['numeric_value'], df['phrase'] = zip(*df['employLengthStatus'].apply(extract_numeric_and_phrase))

display(df)


# In[9]:


# Changed the value from null to 0 for all of the entries where no indicated information on number of years working 
df['numeric_value'] = df.apply(lambda x: 0 if pd.isnull(x['numeric_value']) else x['numeric_value'], axis = 1)
df['phrase'] = df.apply(lambda x: 'Unknown' if pd.isnull(x['phrase']) else x['phrase'], axis = 1)


# In[10]:


df


# In[11]:


# HEAT MAP 0.0 is representing all the entries where people didn't specify how many years they have been at the firm


# Each cell in the heatmap contains a value that represents the proportion (normalized count) of occurrences of a particular 
#"Rating" for a specific combination of "numeric_value" and "phrase".


# Group the data by the combination of numeric_value and phrase
grouped = df.groupby(['numeric_value', 'phrase'])

# Calculate the distribution of ratings within each group
rating_distribution = grouped['rating'].value_counts(normalize=True).unstack()

# Plotting the distribution using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(rating_distribution, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Rating Distribution by Numeric Value and Phrase')
plt.xlabel('Rating')
plt.ylabel('Numeric Value and Phrase')
plt.show()


# In[12]:


stop=set(stopwords.words('english'))

corpus=[]
new= df['conwords'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

from collections import defaultdict
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1


# In[13]:


top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)
plt.bar(x,y)
plt.title('Top Stop Words for Con Words')


# In[14]:


def plot_top_non_stopwords_barchart(text, excluded_words=[], num_words = 14):
    stop = set(stopwords.words('english'))
    stop.update(excluded_words)
    
    new = text.str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]

    counter = Counter(corpus)
    most = counter.most_common()
    filtered_most = [(word, count) for word, count in most if word not in stop]

    x = [word for word, _ in filtered_most[:num_words]]
    y = [count for _, count in filtered_most[:num_words]]
    
    sns.barplot(x=y, y=x)

excluded_stopwords = ['-', 'The']


plot_top_non_stopwords_barchart(df['conwords'], excluded_words=excluded_stopwords)
plt.title('Top Non Stopword Con Words')
plt.xlabel('Frequency')
plt.ylabel('Con Word')
plt.show()


# In[15]:


def plot_top_2ngrams_barchart(text, excluded_words=[], n=2, num_items=10):
    stop = set(stopwords.words('english'))
    stop.update(excluded_words)

    new = text.str.split()
    new = new.values.tolist()
    corpus = [' '.join([word for word in i if word not in stop]) for i in new]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:num_items]

    top_n_bigrams = _get_top_ngram(corpus, n)
    x, y = map(list, zip(*top_n_bigrams))
    sns.barplot(x=y, y=x)

excluded_stopwords = ['reading', 'cons', 'none', 'na']  # Separating some words to do a quick temp fix
desired_num_items = 10

plot_top_2ngrams_barchart(df['conwords'], excluded_words=excluded_stopwords, n=2, num_items=desired_num_items)
plt.title('Top 2-grams in Con Words')
plt.xlabel('Frequency')
plt.ylabel


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_3ngrams_barchart(text, excluded_words=[], n=3, num_items=10):
    stop = set(stopwords.words('english'))
    stop.update(excluded_words)

    new = text.str.split()
    new = new.values.tolist()
    corpus = [' '.join([word for word in i if word not in stop]) for i in new]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        
        # Filter out n-grams containing excluded words
        words_freq = [(word, count) for word, count in words_freq if not any(excluded_word in word.split() for excluded_word in excluded_words)]
        
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:num_items]

    top_n_trigrams = _get_top_ngram(corpus, n)
    x, y = map(list, zip(*top_n_trigrams))
    sns.barplot(x=y, y=x)

excluded_words = ['none', 'na', 'nothing', 'reading', 'continue', 'good', 'great', ]
desired_num_items = 10

plot_top_3ngrams_barchart(df['conwords'], excluded_words=excluded_words, n=3, num_items=desired_num_items)
plt.title('Top 3-grams in Con Words')
plt.xlabel('Frequency')
plt.ylabel('3-gram')
plt.show()


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_4ngrams_barchart(text, excluded_words=[], n=4, num_items=10):
    stop = set(stopwords.words('english'))
    stop.update(excluded_words)

    new = text.str.split()
    new = new.values.tolist()
    corpus = [' '.join([word for word in i if word not in stop]) for i in new]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        
        # Filter out n-grams containing excluded words
        words_freq = [(word, count) for word, count in words_freq if not any(excluded_word in word.split() for excluded_word in excluded_words)]
        
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:num_items]

    top_n_trigrams = _get_top_ngram(corpus, n)
    x, y = map(list, zip(*top_n_trigrams))
    sns.barplot(x=y, y=x)

excluded_words = ['none', 'na', 'nothing', 'reading', 'continue', 'good' ]
desired_num_items = 10

plot_top_4ngrams_barchart(df['conwords'], excluded_words=excluded_words, n=4, num_items=desired_num_items)
plt.title('Top 4-grams in Con Words')
plt.xlabel('Frequency')
plt.ylabel('4-gram')
plt.show()

