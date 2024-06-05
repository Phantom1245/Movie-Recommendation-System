#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import pandas as pd


# In[103]:


#extracting data

movies = pd.read_csv('tmdb_5000_movies.csv')   
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[104]:


movies.head(3)


# In[105]:


credits.head()


# In[106]:


#merging two dataframes

movies = movies.merge(credits, on = 'title')
movies.head(3)


# In[107]:


movies['original_language'].value_counts()
# most pf the movies are of english language, so the original language column isn't going to affect our prediction on a large scale.


# In[108]:


movies.info()


# In[109]:


movies = movies[['id','genres','keywords','overview','title','cast','crew']]


# In[110]:


movies.head(3)


# In[111]:


movies.isnull().sum()  #checking if there are any null values present


# In[112]:


movies.dropna(subset=['overview'],inplace=True)  #dropping all the null values


# In[113]:


movies.duplicated().sum() #checking if any duplicate values are present


# In[114]:


movies.iloc[0].genres


# In[115]:


#Should convert the string list into normal list using ast library
import ast
ast.literal_eval(movies.iloc[0].genres)


# In[116]:


#to extract only the genres

def dict_to_list(obj):
  l=[]
  for i in ast.literal_eval(obj):
    l.append(i['name'])
  return l


# In[117]:


movies['genres'] = movies['genres'].apply(dict_to_list)  #applies the function to whole dataset at a time


# In[118]:


movies.iloc[0].genres


# In[119]:


movies['keywords'] = movies['keywords'].apply(dict_to_list)


# In[120]:


#to extract top 3 casts

def dict_to_list_cast(obj):
  l=[]
  c=0
  for i in ast.literal_eval(obj):
    if(c==3):
      break
    l.append(i['name'])
    c+=1
  return l


# In[121]:


movies['cast'] = movies['cast'].apply(dict_to_list_cast)


# In[122]:


movies.head()


# In[123]:


movies.iloc[0].crew


# In[124]:


#for extracting only director

def dict_to_list_crew(obj):
  l=[]
  for i in ast.literal_eval(obj):
    if(i['job']=='Director'):
      l.append(i['name'])
  return l


# In[125]:


movies['crew'] = movies['crew'].apply(dict_to_list_crew)


# In[126]:


movies.head()


# In[127]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())  #converting overview into a list


# In[128]:


movies.head()


# In[129]:


#to remove spaces between words if any present

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[130]:


movies.head()


# In[131]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[132]:


movies.head()


# In[133]:


movies['tags'] = movies['tags'].apply(lambda x:" ".join(x))


# In[134]:


new_movies = movies[['id','title','tags']]


# In[135]:


new_movies.head()


# In[136]:


new_movies['tags'] = new_movies['tags'].apply(lambda x:x.lower())


# The below library is imported to remove the various tenses of a single verb which will be useful in later parts.

# In[137]:


import nltk


# In[138]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[139]:


def stem(text):
  l=[]
  for i in text.split():
    l.append(ps.stem(i))
  return " ".join(l)


# In[140]:


print(stem('loving'))
print(stem('loved'))
print(stem('love'))


# In[141]:


new_movies['tags'] = new_movies['tags'].apply(stem)


# In[142]:


new_movies.iloc[0].tags


# In[143]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[144]:


vectors = cv.fit_transform(new_movies['tags']).toarray()


# In[145]:


vectors.shape


# In[146]:


vectors[0]


# In[147]:


np.set_printoptions(threshold = np.inf)  #to print full numpy array without truncation
print(cv.get_feature_names_out())


# In[148]:


from sklearn.metrics.pairwise import cosine_similarity


# In[149]:


similarity = cosine_similarity(vectors)


# In[154]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[157]:


def recommend(movie):
    movie_index = new_movies[new_movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_movies.iloc[i[0]].title)


# In[158]:


recommend('Avatar')


# In[ ]:




