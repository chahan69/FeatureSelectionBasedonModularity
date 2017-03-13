#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,io
import numpy as np
from time import time
from random import sample
from math import sqrt, log2

print(np.get_include())

class Newman(object):
	"""Fast Newman algorithm for sparce data"""
	def __init__(self, data, distanse = "cosine", profile=True):

		self.time = 0
		self.distanse = distanse
		self.profile = profile

		self.data = np.matrix(data, dtype = float)

		self.fftime = 0
		self.eiitime= 0
		self.as2time= 0
		self.a2stime= 0


	def multi(self,i):

		t0 = time()
		index = self.mask.A[:,i]
		self.fftime += time()-t0

		t0 = time()
		eii = np.compress(index, np.compress(index, self.A, axis=0) , axis=1).sum()
		self.eiitime += time()-t0

		t0 = time()
		as2 = self.a[self.mask[:,i].T].sum() ** 2
		self.as2time += time()-t0

		t0 = time()
		a2s = self.a2[self.mask[:,i].T].sum()
		self.a2stime += time()-t0

		return(eii - as2 + a2s)

	def ranking(self, sampling = "All"):

		print("Feature Seelection Based on Modularity :D")
		print("sampling  :",sampling)

		# サンプリング
		n = self.data.shape[0]
		if sampling == "sqrt":
			self.data = np.matrix(sample(self.data.tolist(), int(sqrt(n))))
		elif sampling == "sqrtlog":
			self.data = np.matrix(sample(self.data.tolist(), int(sqrt(n)*log2(n))))

		self.mask = self.data.astype(bool)

		print("Instance  :",self.data.shape[0])
		print("Features  :",self.data.shape[1])
		sys.stdout.flush()

		# 正規化
		def normalize():
			if self.profile:
				print("normalize : ",end="")
			t0 = time()

			for i in range(self.data.shape[0]):
				self.data[i] = np.divide(self.data[i], np.linalg.norm(self.data[i]))

			t1 = time() - t0
			self.time += t1
			if self.profile:
				print(t1,"sec")
				sys.stdout.flush()

		# 隣接行列の計算
		def adjacent():
			if self.profile:
				print("Ad.matrix : ",end="")
			t0 = time()

			self.A = self.data.dot(self.data.T)
			for i in range(len(self.data)):
				self.A[i,i] = 0
			self.a = np.array(self.A.sum(0)) # 早いほうで
			self.MM = self.a.sum()
			self.a = np.divide(self.a,self.MM)
			self.A = np.divide(self.A,self.MM)

			t1 = time() - t0
			self.time += t1
			if self.profile:
				print(t1,"sec")
				sys.stdout.flush()

		# スコアの計算
		def socoring():
			if self.profile:
				print("socoring  : ",end="")
			# t0 = time()

			self.a2 = self.a ** 2

			# p = Pool(12)
			# self.rank = p.map(self.multi,[i for i in range(self.data.shape[1])], 100000)
			self.rank = [self.multi(i) for i in range(self.data.shape[1])]

			t1 = self.fftime+self.eiitime+self.as2time+self.a2stime
			self.time += t1
			if self.profile:
				print(t1,"sec")
				print("TotalTime :",self.time,"sec\n")
				sys.stdout.flush()
				# print("density   :",self.A.nnz/self.A.shape[0]**2)

		normalize()
		adjacent()
		socoring()
		# print("ff :",self.fftime)
		# print("eii:",self.eiitime)
		# print("as2:",self.as2time)
		# print("a2s:",self.a2stime)

		return(self.rank)

		


if __name__ == '__main__':
	# d=[\
	# [1 ,0 ,0 ],\
	# [1 ,1 ,0 ],\
	# [1 ,0 ,0 ],\
	# [0 ,0 ,1 ],\
	# [0 ,0 ,1 ],\
	# [0 ,1 ,1 ]\
	# ]

	# d=[\
	# [1 ,0 ,0 ,0],\
	# [1 ,0 ,0 ,0],\
	# [1 ,1 ,0 ,1],\
	# [0 ,1 ,1 ,1],\
	# [0 ,0 ,1 ,0],\
	# [0 ,0 ,1 ,0]\
	# ]

	d=[[1,0,0,0],
	   [1,0,0,0],
	   [1,1,0,0],
	   [0,1,1,0],
	   [0,0,1,0],
	   [0,0,1,1],
	   [0,0,0,1]]

	n=Newman(d)
	data=d
	print(n.ranking(sampling="all"))
