
# coding: utf-8

# In[125]:




import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse.linalg as sla

#a)Load all images in data Matrix X
X=np.zeros((165,77760))
Type=['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
for i in range(15):
    for j in range(11):
        filename="yalefaces\subject{:0>2d}.{}".format(i+1,Type[j])
        X[i*11+j]=plt.imread(filename).reshape((77760,))


# In[227]:


#b)Compute the mean face and center the whole X
meanface=np.sum(X,axis=0)/165
cX=X-meanface

#c)Compute SVD and find V according to eigenvector of 60 largest eigenvalue
u, s, vt = sla.svds(cX,k=60)

#d)find 60 dimensional representation Z
Z=np.dot(cX,vt.T)

#e)reconstruct X
rX=meanface.T+np.dot(Z,vt)

#e)report error
error=np.zeros((12,))
for i in range(165):
    error[11]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[11]=error[11]/165


# In[228]:


pNo=2
plt.imshow(X[pNo].reshape((243,320)), cmap='gray'), plt.title("origin")


# In[229]:


plt.imshow(rX[pNo].reshape((243,320)), cmap='gray'), plt.title("p=60")
print("Reconstruction Error:",error[11])


# In[230]:


u, s, vt = sla.svds(cX,k=5)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[0]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[0]=error[0]/165
print("Reconstruction Error:",error[0])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'), plt.title("p=5")


# In[231]:


u, s, vt = sla.svds(cX,k=10)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[1]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[1]=error[1]/165
print("Reconstruction Error:",error[1])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'), plt.title("p=10")


# In[232]:


u, s, vt = sla.svds(cX,k=15)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[2]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[2]=error[2]/165
print("Reconstruction Error:",error[2])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'), plt.title("p=15")


# In[233]:


u, s, vt = sla.svds(cX,k=20)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[3]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[3]=error[3]/165
print("Reconstruction Error:",error[3])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'),plt.title("p=20")


# In[234]:


u, s, vt = sla.svds(cX,k=25)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[4]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[4]=error[4]/165
print("Reconstruction Error:",error[3])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'),plt.title("p=25")


# In[235]:


u, s, vt = sla.svds(cX,k=30)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[5]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[5]=error[5]/165
print("Reconstruction Error:",error[5])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'),plt.title("p=30")


# In[236]:


u, s, vt = sla.svds(cX,k=35)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[6]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[6]=error[6]/165
print("Reconstruction Error:",error[6])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'),plt.title("p=35")


# In[237]:


u, s, vt = sla.svds(cX,k=40)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[7]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[7]=error[7]/165
print("Reconstruction Error:",error[7])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'),plt.title("p=40")


# In[238]:


u, s, vt = sla.svds(cX,k=45)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[8]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[8]=error[8]/165
print("Reconstruction Error:",error[8])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'),plt.title("p=45")


# In[239]:


u, s, vt = sla.svds(cX,k=50)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[9]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[9]=error[9]/165
print("Reconstruction Error:",error[9])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'),plt.title("p=50")


# In[240]:


u, s, vt = sla.svds(cX,k=55)
Z=np.dot(cX,vt.T)
rX=meanface.T+np.dot(Z,vt)
for i in range(165):
    error[10]+=np.dot(X[i]-rX[i],(X[i]-rX[i]).T)
error[10]=error[10]/165
print("Reconstruction Error:",error[10])
plt.imshow(rX[pNo].reshape((243,320)),cmap='gray'),plt.title("p=55")


# In[223]:


p=[5,10,15,20,25,30,35,40,45,50,55,60]
plt.plot(p,error), plt.xlabel("p"), plt.ylabel("error")

