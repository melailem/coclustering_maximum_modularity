#!/usr/bin/env python

import sys
import getopt
import re
import os,glob
from math import * 
import numpy 
from numpy import *
from collections import *


import marshal
import cPickle
import pickle
from sklearn.feature_extraction.text import *
import matplotlib.pyplot as plt

#import community
import igraph
from igraph import *
from random import randint




##### donnees classic30 (classic 3 avec 30 documents) 
import scipy.io
mat = scipy.io.loadmat("/users/lipade2/mailem/pylouvain/classic30.mat")

print mat.keys()

matrice = mat['dtm']



print matrice.shape

rows=range(0,matrice.shape[0])
cols=range(0,matrice.shape[1])





degres_rows=[] #### degres lignes
degres_cols=[] #### degres colonnes

for j in range(0,len(cols)):
	d=0
	for i in range(0,len(rows)):
		if matrice[i,j] > 0 :
			d=d+1
	degres_cols.append(d)


for i in range(0,len(rows)):
	d=0
	for j in range(0,len(cols)):
		if matrice[i,j] > 0 :
			d=d+ 1
	degres_rows.append(d)


print len(degres_cols)
print len(degres_rows)



m=sum(degres_cols) #### nombre d aretes


matrice2=numpy.zeros((len(rows),len(cols)))



for i in range(0,len(rows)):
	for j in range(0,len(cols)):
		if matrice[i,j] > 0:
			matrice2[i,j]=1



#####################################################coclustering====================================================================================================================

####initialisation

clusters=3 ###### nombre de clusters 

#Z=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2] ####partition initiale de Z random

Z=[]


for i in range(0,len(rows)):
	Z.append(randint(0,clusters-1))

print Z



W=[]
for i in range(0,len(cols)):
	W.append(-1)

change=True

news=[]
oldmodu=float("-inf")


while (change):
	change=False


	degres_k=[0] * clusters
	
	for j in range(0,len(Z)):
		degres_k[Z[j]]=degres_k[Z[j]]+degres_rows[j]
	


	for i in range(0,len(cols)):
		
		max=float("-inf")
		for k in range(0,clusters):
			a=0
			nbr_k=0
			for j in range(0,len(Z)):
				if Z[j] == k :
					a=a+matrice2[j,i]
					nbr_k=nbr_k+1
			nbr_k=nbr_k/(len(rows)*1.)
			Pik=(degres_cols[i] * degres_k[k]) / (1.*m)
			modu=(a-(Pik))
			if modu > max :
				new=k
				max= modu
		if new != W[i]:
			change=True
			W[i] = new


	#print max

	degres_l=[0] * clusters
	for j in range(0,len(W)):
		degres_l[W[j]]=degres_l[W[j]]+degres_cols[j]
	

	for r in range(0,len(rows)):
		max=float("-inf")
		for l in range(0,clusters):
			a=0
			nbr_l=0
			for j in range(0,len(W)):
				if W[j] == l :
					a=a+matrice2[r,j]
					nbr_l=nbr_l+1
			nbr_l=nbr_l/(len(cols)*1.)
			Prl=(degres_rows[r] * degres_l[l]) / (1.*m)
			modu=(a - (Prl))
			if modu > max :
				new=l
				max=modu
		if new != Z[r]   :
			change=True
			Z[r] = new
         
	
	#if oldmodu < max+x : 
        	#oldmodu = max+x
        #else :
		#change = False

	Q=0
	for p in range(0,len(rows)):
		for q in range(len(rows),len(rows)+len(cols)):
			if Z[p]==W[q-len(rows)]:
				ppq=(degres_rows[p]*degres_cols[q-len(rows)])/(m*1.)
				Q=Q+(matrice2[p,q-len(rows)]-ppq)

	print Q/(m*1.)
	news.append(Q/(m*1.))
	

plt.plot(news)
plt.ylabel('some numbers')
plt.show()

print Z
print W

########################################################################reorganisation des lignes et des colonnes et affichage de la matrice


ow=[]
oz=[]

for j in range(0,clusters):
	for i in range(0,len(W)):
		if W[i] == j:
			ow.append(i)


for j in range(0,clusters):
	for i in range(0,len(Z)):
		if Z[i] == j:
			oz.append(i)


j=numpy.array(oz)
a=matrice2[j, :]


i = numpy.array(ow)
b=a[:, i]






plt.imshow(b, interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5,10.5,0.5,10.5))
plt.colorbar()
plt.show()









