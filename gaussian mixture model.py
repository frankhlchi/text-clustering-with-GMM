#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import imageio
from matplotlib import pylab as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from numpy import linalg as LA
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal


# ### Clustering for text analysis
# 
# To obtain the features, we performed the following transformation. First, we computed perdocument smoothed word frequencies. Second, we took the log of those frequencies. Finally, we centered the per-document log frequencies to have zero mean. Cluster the documents using k-means and various values of k (go up to at least k = 20).
# Select a value of k.

# In[16]:


def read_text(filename):
    line_array = []
    f = open(filename, 'r')
    for line in f:
        line_array.append(line.strip())
    f.close()
    return line_array 

word_array = np.array(read_text("text/science2k-vocab.txt"))
title_array = np.array(read_text("text/science2k-titles.txt"))
doc_word = np.load("text/science2k-doc-word.npy")
word_doc = np.load("text/science2k-word-doc.npy")


# In[17]:


word_doc.shape


# Question 2(a)

# In[18]:


mean_doc_word = np.mean(doc_word, axis=0)
max_num = 20
k_means = {'center':[], 'labels' :[]}
for num_ in range(2, max_num + 1):
    model = KMeans(n_clusters = num_).fit(doc_word)
    k_means['center'].append(model.cluster_centers_)
    k_means['labels'].append(model.labels_)


# In[19]:


#report the top 10 words of each cluster in order of the largest positive distance
#from the average value across all data. 

with open("top_words_a.txt","w") as f:
    for i in range(len( k_means['center'])):
        f.writelines('k = %i \n'%(i +2))
        print ('k = %i'%(i +2))
        counter = 1 
        for j in [(i -  mean_doc_word).argsort()[-10:][::-1] for i in k_means['center'][i]]:
            print ('cluster %i'%(counter))
            print ([word_array[ind] for ind in j ])
            f.writelines(' cluster %i -'%(counter))
            f.writelines(str([word_array[ind] for ind in j ]))
            counter +=1
        f.writelines('\n')


# In[20]:


# Report the top ten documents that fall closest to each cluster center
with open("top_docs_a.txt","w") as f:
    for i in range(len( k_means['center'])):
        print ('k = %i'%(i +2))
        f.writelines('k = %i \n'%(i +2))
        counter = 1 
        for m, n in zip(k_means['center'][i],range(len(k_means['center'][i]))):
            #select top 10 docs in the coressponding cluster
            doc_word_ = doc_word[k_means['labels'][i] == n,:]
            for j in [np.array([LA.norm(m - doc_word_[doc,:]) for doc in range(doc_word_.shape[0])])                      .argsort()[:10]]:
                print ('cluster %i'%(counter))
                f.writelines(' cluster %i -'%(counter))
                for ind in j:
                    print (title_array[k_means['labels'][i] == n][ind])
                    f.writelines(' %s '%title_array[k_means['labels'][i] == n][ind])
                counter +=1
        print () 
        f.writelines('\n')


# Question 2(b)

# In[21]:


mean_word_doc = np.mean(word_doc, axis=0)
max_num = 20
k_means_wd = {'center':[],'labels':[]}
for num_ in range(2, max_num + 1):
    model = KMeans(n_clusters = num_).fit(word_doc)
    k_means_wd['center'].append(model.cluster_centers_)
    k_means_wd['labels'].append(model.labels_)


# In[22]:


with open("top_doc_b.txt","w") as f:
    for i in range(len( k_means['center'])):
        print ('k = %i'%(i + 2))
        f.writelines('k = %i \n'%(i +2))
        counter = 1 
        for j in [(m -  mean_word_doc).argsort()[-10:][::-1] for m in k_means_wd['center'][i]]:
            print ('cluster %i'%(counter))
            f.writelines(' cluster %i -'%(counter))
            for ind in j:
                print (title_array[ind]) 
                f.writelines(' %s '%title_array[ind])
            counter +=1
        print ()
        f.writelines('\n')


# In[23]:


with open("top_word_b.txt","w") as f:
    for i in range(len( k_means_wd['center'])):
        print ('k = %i'%(i +2))
        f.writelines('k = %i \n'%(i +2))
        counter = 1 
        for m,n in zip(k_means_wd['center'][i],range(len(k_means_wd['center'][i]))):
            #select top 10 words in the coressponding cluster
            word_doc_ =  word_doc[k_means_wd['labels'][i] == n,:]
            for j in [np.array([LA.norm(m - word_doc_[doc,:]) for doc in range(word_doc_.shape[0])])                      .argsort()[:10]]:
                print ('cluster %i'%(counter))
                print ([word_array[k_means_wd['labels'][i] == n][ind] for ind in j ])
                f.writelines(' cluster %i -'%(counter))
                f.writelines(str([word_array[k_means_wd['labels'][i] == n][ind] for ind in j ]))
                counter +=1
        print ()
        f.writelines('\n')


# ### EM algorithm and implementation

# (b) Download the Old Faithful Geyser Dataset. The data file contains 272 observations of (eruption time, waiting time). Treat each entry as a 2 dimensional feature vector. Parse and plot all
# data points on 2-D plane.

# In[24]:


faithful_data = pd.read_csv('faithful.csv')[['eruptions','waiting']]
faithful_data.head(5)


# In[25]:


plt.scatter(x=faithful_data.eruptions,y = faithful_data.waiting )
plt.xlabel('eruptions time')
plt.ylabel('waiting time')


# (c) Implement a bimodal GMM model to fit all data points using EM algorithm. Explain the reasoning behind your termination criteria. 

# In[26]:


#self-implemeted GMM
class Bimodel_GMM():
    
    def __init__(self, n_clusters = 2, random_init = False,kmeans_init =False,plot = False,ramdom_state_kmean=0):
        self.n_clusters = n_clusters
        self.mu_10 = None
        self.mu_11 = None
        self.sigma_10 = None
        self.sigma_11 = None
        self.mu_20 = None
        self.mu_21 = None
        self.sigma_20 = None
        self.sigma_21 = None
        self.pi = None
        self.mu_changes = []
        self.random_init = random_init
        self.kmeans_init = kmeans_init
        self.plot = plot
        self.ramdom_state_kmeans = ramdom_state_kmean
        return 
    
    def initialize_para(self,data):
        #initial parameter 
        #default initialization policy
        factor = 0.5
        #random initialization policy
        if self.random_init:
            factor = np.random.rand(1)[0]
            print ('randomly initalized with factor %f'%factor)

        self.mu_10 = np.mean(data[:,0]) + np.std(data[:,0]) * 3 * factor
        self.mu_20 = np.mean(data[:,0]) - np.std(data[:,0]) * 3 * factor
        self.mu_11 = np.mean(data[:,1]) + np.std(data[:,1]) * 3 * factor
        self.mu_21 = np.mean(data[:,1]) - np.std(data[:,1]) * 3 * factor
        self.sigma_10 = np.var(data[:,0]) 
        self.sigma_11 = np.var(data[:,1]) 
        self.sigma_20 = np.var(data[:,0]) 
        self.sigma_21 = np.var(data[:,1])
        self.pi  = 0.5
        #k-means initialization policy
        if self.kmeans_init:
            print ('initalized with k-means')
            label = KMeans(n_clusters = 2, random_state = self.ramdom_state_kmeans).fit(data).predict(data)
            #use maximum likelihood to extimate patameter
            self.mu_10 = np.mean(data[label == 1][:,0]) 
            self.mu_11 = np.mean(data[label == 1][:,1]) 
            self.mu_20 = np.mean(data[label == 0][:,0])
            self.mu_21 = np.mean(data[label == 0][:,1]) 
            self.sigma_10 = np.var(data[label == 1][:,0])
            self.sigma_11 = np.var(data[label == 1][:,1])
            self.sigma_20 = np.var(data[label == 0][:,0])
            self.sigma_21 = np.var(data[label == 0][:,1])
            self.pi = len(data[label == 1])/len(data)
        return
    
    def reassignment(self, data):
        assignment = []
        mu_1 = np.array([self.mu_10,self.mu_11])
        mu_2 = np.array([self.mu_20,self.mu_21])
        sigma_1 = np.diag([self.sigma_10,self.sigma_11])
        sigma_2 = np.diag([self.sigma_20,self.sigma_21])        
        for obs in data:
            prob_1 = multivariate_normal.pdf(obs, mean=mu_1, cov=sigma_1, allow_singular=True)
            prob_2 = multivariate_normal.pdf(obs, mean=mu_2, cov=sigma_2, allow_singular=True)
            prob_cluster_1 = (prob_1 * self.pi)/(prob_1 * self.pi + prob_2 * (1- self.pi))
            assignment.append(prob_cluster_1)
       
        if  self.plot:
            plt.scatter(data[:, 0], data[:, 1] , c= np.round(np.array(assignment)), s=40, cmap='viridis')
            plt.show()
        return np.array(assignment)
    
    def mean_update(self, assignment):
        self.mu_10 = np.sum(assignment*self.data[:,0])/np.sum(assignment)
        self.mu_11 = np.sum(assignment*self.data[:,1])/np.sum(assignment)
        self.mu_20 = np.sum((1- assignment)*self.data[:,0])/np.sum(1- assignment)
        self.mu_21 = np.sum((1- assignment)*self.data[:,1])/np.sum(1- assignment)
        return  
    
    def var_update(self, assignment):
        self.sigma_10 = np.sum(((self.data[:,0] - self.mu_10)**2)*assignment)/np.sum(assignment)
        self.sigma_11 = np.sum(((self.data[:,1] - self.mu_11)**2)*assignment)/np.sum(assignment)
        self.sigma_20 = np.sum(((self.data[:,0] - self.mu_20)**2)*(1- assignment))/np.sum(1- assignment)
        self.sigma_21 = np.sum(((self.data[:,1] - self.mu_21)**2)*(1- assignment))/np.sum(1- assignment)
        return
    
    def pi_update(self, assignment):
        self.pi = np.sum(assignment)/len(assignment)
    
    def expectation_max(self, data, thres = 1e-5, max_iteration =1000):
        counter = 1
        pre_mu = np.array([self.mu_10, self.mu_11, self.mu_20, self.mu_21,self.pi]) 
        for i in range(max_iteration):
            self.mu_changes.append(pre_mu)           
            assign = self.reassignment(data)
            self.mean_update(assign)
            self.var_update(assign)
            self.pi_update(assign)
            
            if np.linalg.norm(np.array([self.mu_10, self.mu_11, self.mu_20, self.mu_21,self.pi]) - pre_mu) < thres :
                print ('stop at %i iterations'%(i+1))
                return counter
            counter +=1
            pre_mu = np.array([self.mu_10, self.mu_11, self.mu_20, self.mu_21,self.pi])
        return counter
        
    def fit(self, data):
        self.data = data
        self.initialize_para(data)
        num_of_iter = self.expectation_max(data, max_iteration =1000)
        return num_of_iter
    
    def predict(self, data):
        prediction = np.round(self.reassignment(data))
        return prediction 


# In[27]:


GMM_model = Bimodel_GMM(n_clusters = 2,kmeans_init=False,random_init=False,plot=False)
number_of_iter = GMM_model.fit(faithful_data.values)


# In[28]:


np.array(GMM_model.mu_changes)


# In[29]:


fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(np.array(GMM_model.mu_changes)[:,0],                np.array(GMM_model.mu_changes)[:,1])
plt.xlabel('mu_10')
plt.ylabel('mu_11')
plt.title('mean vector of the cluster 1 with label of number of iteration')
for i in range(number_of_iter):
    ax.annotate(range(1,number_of_iter +1)[i],                 (np.array(GMM_model.mu_changes)[:,0][i], np.array(GMM_model.mu_changes)[:,1][i]))


# In[30]:


fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(np.array(GMM_model.mu_changes)[:,2],                np.array(GMM_model.mu_changes)[:,3])
plt.xlabel('mu_20')
plt.ylabel('mu_21')
plt.title('mean vector of the cluster 2 with label of number of iteration')
for i in range(number_of_iter):
    ax.annotate(range(1,number_of_iter +1)[i],                 (np.array(GMM_model.mu_changes)[:,2][i], np.array(GMM_model.mu_changes)[:,3][i]))


# In[31]:


num_of_iteration = [] 
for i in range(50):
    GMM_model = Bimodel_GMM(n_clusters = 2,kmeans_init=False,random_init=True)
    num = GMM_model.fit(faithful_data.values)
    num_of_iteration.append(num)


# In[32]:


plt.figure(figsize=(8,4))
plt.hist(np.array(num_of_iteration),bins=5)
plt.title('distribution of number of iterations needed for convergence')


# In[33]:


print ('average num of iteratons', np.mean(num_of_iteration))


# (d) Repeat the task in (c) but with the initial guesses of the parameters generated from the following process:
# • Run a k-means algorithm over all the data points with K = 2 and label each point with
# one of the two clusters.
# • Estimate the first guess of the mean and covariance matrices using maximum likelihood
# over the labeled data points.

# In[34]:


# initialzied with k-means
GMM_model_kmeans = Bimodel_GMM(n_clusters = 2,kmeans_init= True)
number_of_iter = GMM_model_kmeans.fit(faithful_data.values)


# In[35]:


num_of_iteration_kmeans = [] 
for i in range(50):
    GMM_model_kmeans = Bimodel_GMM(n_clusters = 2,kmeans_init= True,ramdom_state_kmean = i )
    number_of_iter = GMM_model_kmeans.fit(faithful_data.values)
    num_of_iteration_kmeans.append(number_of_iter)


# In[36]:


plt.figure(figsize=(8,4))
plt.hist(np.array(num_of_iteration_kmeans),bins=5)
plt.title('distribution of number of iterations needed for convergence (k-means)')


# In[37]:


print ('average num of iteratons', np.mean(num_of_iteration_kmeans))


# In[38]:


fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(np.array(GMM_model_kmeans.mu_changes)[:,0],                np.array(GMM_model_kmeans.mu_changes)[:,1])
plt.xlabel('mu_10')
plt.ylabel('mu_11')
plt.title('mean vector of the cluster 1 with label of number of iteration (k-mean initialized)')
for i in range(number_of_iter):
    ax.annotate(range(1,number_of_iter +1)[i],                 (np.array(GMM_model_kmeans.mu_changes)[:,0][i], np.array(GMM_model_kmeans.mu_changes)[:,1][i]))


# In[39]:


fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(np.array(GMM_model_kmeans.mu_changes)[:,2],                np.array(GMM_model_kmeans.mu_changes)[:,3])
plt.xlabel('mu_20')
plt.ylabel('mu_21')
plt.title('mean vector of the cluster 2 with label of number of iteration (k-mean initialized)')
for i in range(number_of_iter):
    ax.annotate(range(1,number_of_iter +1)[i],                 (np.array(GMM_model_kmeans.mu_changes)[:,2][i], np.array(GMM_model_kmeans.mu_changes)[:,3][i]))


# In[ ]:





# In[ ]:




