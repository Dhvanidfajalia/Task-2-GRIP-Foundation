#!/usr/bin/env python
# coding: utf-8

# # GRIP: THE Sparks Foundation
# 
# ## Data Science and Business Analytics Intern
# 
# ## Author: Dhvani Fajalia
# 
# ###  Task 2: Prediction Using Unsupervised ML
# 

# #### In this we have to predict the optimum number of cluster and represent it visually using dataset named Iris.

# ####  Step 1: Importing Libraries

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# #### Step 2: Reading data

# In[13]:


data_iris = datasets.load_iris()
df = pd.DataFrame(data_iris.data, columns = data_iris.feature_names)
df.head()


# In[15]:


df.columns = [i.split('(')[0] for i in df.columns]
df


# In[17]:


print(df.shape)
df.describe()


# ####  Step 3: Now we can find optimum number of clusters for k-means classification 

# In[18]:


x = df.iloc[:, :].values
x


# In[22]:


from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# #### Step 4: Visualizing the line graph for Elbow 

# In[23]:


plt.style.use('ggplot')
plt.plot(range(1, 11), wcss, color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# #### Now we can see that number of cluster will be '3' 

# In[24]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[29]:


# Visualizing the clusters 

plt.style.use('seaborn')
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa', alpha=0.5)
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour', alpha=0.5)
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], 
            s = 100, c = 'green', label = 'Iris-virginica', alpha=0.5)

# Plotting the centroid of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'cyan', label = 'Centroids')
plt.legend()
plt.show()


# ## Thus the above plot shows the Centroid for every category of iris plant.

# ## Thank you Sparks foundation for giving this task!

# ### It has helped me to understand unsupervised learning in a better way.

# In[ ]:




