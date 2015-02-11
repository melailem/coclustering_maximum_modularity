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
import sklearn
import matplotlib.pyplot as plt

#import community
import igraph
from igraph import *
from random import randint
import itertools
from scipy.io import loadmat, savemat
from sklearn.metrics import confusion_matrix
from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.metrics import accuracy_score
import scipy



mat=loadmat("/users/lipade2/mailem/pylouvain/A_tester/cstr.mat")

print mat.keys()


matrice = mat['fea']
print matrice.shape


labels=mat['gnd']

labels=labels.tolist()
labels = list(itertools.chain.from_iterable(labels))


label=[]

for l in labels:
	label.append(l-1)

labels=label

###
label=mat['gnd']

label=label.tolist()
label = list(itertools.chain.from_iterable(label))


label=[x-1 for x in label]





rows=range(0,matrice.shape[0])
cols=range(0,matrice.shape[1])


matrice=sklearn.preprocessing.binarize(matrice)




degres_rows=matrice.sum(axis=1).tolist()
degres_cols=matrice.sum(axis=0).tolist()

"""
numpy.savetxt("/users/lipade2/mailem/matrice_non_sparse/cstr.txt", matrice, delimiter='\t')

sys.exit()
"""

m=sum(degres_cols)
print m




#####################################################coclustering====================================================================================================================

####initialisation



clusters=4 ###### nombre de clusters 

Acc=[]

for it in range(0,100): ##### executer lalgo 30 fois
	print "iteration "+ str(it) + "..."





	Z=[]


	for i in range(0,len(rows)):
		Z.append(randint(0,clusters-1))


	"""
	Z=label

	for i in range(0,len(label),2):
		Z[i]=randint(0,clusters-1)
	#print Z

	"""


	W=[]
	for i in range(0,len(cols)):
		W.append(-1)


	
	news=[]

	fil=open("/users/lipade2/mailem/pylouvain/cstr_100_new/resultat_cstr_"+str(it)+".txt","w")
	print >> fil, "initialisation : " + str(Z)


	change=True

	Qmax=float("-inf")
	while (change):


		degres_k=[0] * clusters
	
		for j in range(0,len(Z)):
			degres_k[Z[j]]=degres_k[Z[j]]+degres_rows[j]
	


		for i in range(0,len(cols)):
		
			maxi=float("-inf")
			for k in range(0,clusters):
				a=0
				nbr_k=0
				for j in range(0,len(Z)):
					if Z[j] == k :
						a=a+matrice[j,i]
						nbr_k=nbr_k+1
				nbr_k=nbr_k/(len(rows)*1.)
				Pik=(degres_cols[i] * degres_k[k]) / (1.*m)
				modu=(a-(Pik))*(1/sqrt(nbr_k))
				if modu > maxi :
					W[i]=k
					maxi= modu



		#print max

		degres_l=[0] * clusters
		for j in range(0,len(W)):
			degres_l[W[j]]=degres_l[W[j]]+degres_cols[j]
	

		for r in range(0,len(rows)):
			maxi=float("-inf")
			for l in range(0,clusters):
				a=0
				nbr_l=0
				for j in range(0,len(W)):
					if W[j] == l :
						a=a+matrice[r,j]
						nbr_l=nbr_l+1
				nbr_l=nbr_l/(len(cols)*1.)
				Prl=(degres_rows[r] * degres_l[l]) / (1.*m)
				modu=(a - (Prl))*(1/sqrt(nbr_l))
				if modu > maxi :
					Z[r]=l
					maxi=modu

         
	
		#if oldmodu < max+x : 
        		#oldmodu = max+x
        	#else :
			#change = False

		Q=0
		for p in range(0,len(rows)):
			for q in range(len(rows),len(rows)+len(cols)):
				if Z[p]==W[q-len(rows)]:
					ppq=(degres_rows[p]*degres_cols[q-len(rows)])/(m*1.)
					Q=Q+( (matrice[p,q-len(rows)]-ppq)*(1/sqrt(Z.count(Z[p])))*(1/sqrt(W.count(Z[p]))) )

		Q=Q/(m*1.)
		if (abs(Q-Qmax) < 0.000001) : 
			change=False
		else:
			Qmax=Q
			print Q
			news.append(Q)
	
	"""
	plt.plot(news)
	plt.ylabel('some numbers')
	plt.show()
	"""
	

	####matrice de confusion
	cm=confusion_matrix(labels, Z)

	
	###calcul accuracy a partir de la matrice de confusion (dans le code je dis "purity" mais au fait c'est "accuracy")
	cm=numpy.matrix(cm)
	cm1=cm

	cml=numpy.array(cm).tolist()
	

	cml = list(itertools.chain(*cml))

	total=0
	for i in range(0,clusters):
		if len(cml) != 0:
			ma=max(cml)
			if (ma  in cm1): 
				index = numpy.where(cm1==ma)
				total = total +ma
				cml.remove(ma)
				cm1 = scipy.delete(cm1,index[0][0,0], 0)
				cm1 = scipy.delete(cm1, index[1][0,0], 1)
				cml=numpy.array(cm1).tolist()

				cml = list(itertools.chain(*cml))

	purity=(total)/(len(rows)*1.)
	print "purity ==>" + str(purity)

	Acc.append(purity)


	fil=open("/users/lipade2/mailem/pylouvain/cstr_100_new/resultat_cstr_"+str(it)+".txt","a")
	print >> fil, "Z = "+str(Z) + "\n\n" + "W = " +str(W) + "\n\n" + "news " + str(news) + "\n\n" + "confusion : \n" + str(cm) + "purity \n" + str(purity) 


fil=open("/users/lipade2/mailem/pylouvain/cstr_100_new/final.txt","w")
print >> fil, "Acc = "+str(Acc) + "\n\n" + "moyenne " + str(numpy.mean(Acc)) + "std " + str(numpy.std(Acc))


fil.close()


########################################################################affichage

"""
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
a=matrice[j, :]


i = numpy.array(ow)
b=a[:, i]






plt.imshow(b, interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5,10.5,0.5,10.5))
plt.colorbar()
plt.show()

"""











