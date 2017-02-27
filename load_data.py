import numpy as np
import pickle
import os
from collections import defaultdict
# import time
from datetime import datetime, timezone, timedelta

def loadTime(dir):
	x,y = np.loadtxt(dir+'/Partition6467ProbePoints.csv', dtype=str, delimiter=',', usecols=(0,1), unpack=True)
	x = np.array([fixString(x[i]) for i in range(x.shape[0])])
	y = np.array([datetime.strptime(y[i], "b'%m/%d/20%y %I:%M:%S %p'") for i in range(y.shape[0])])
	return x,y

def fixString(x):
	return x.split("'")[-2]

def loadLink(dir):
	x,y,z,cat = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,1,2,5), unpack=True)
	dist = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=float, delimiter=',', usecols=(3,))
	graph = defaultdict(lambda: [])
	for i in range(y.shape[0]):
		Y = fixString(y[i])
		Z = fixString(z[i])
		C = fixString(cat[i])
		if C == 'F':
			graph[Y].append(Z)
		elif C == 'T':
			graph[Y].append(Z)
		elif C == 'B':
			graph[Y].append(Z)
			graph[Z].append(Y)
		# graph[y[i]].append(z[i])
	return graph

def timeSlots(ids, date_time):
	five_min = timedelta(minutes = 5)
	uni_ids,ind,counts = np.unique(ids, return_index=True, return_counts=True)
	times = defaultdict(lambda: [])
	for i in range(uni_ids.shape[0]):
		count = 0
		while count < counts[i]:
			times[uni_ids[i]].append(date_time[ind[i]+count])
			count += 1
	return times

def loadData(dir):
	pTime = []
	pLatLong = []
	pAlt = []
	pVelocity = []
	lData = []
	if not os.path.exists('pTime.pckl'):
		pTime = np.loadtxt(dir+'/Partition6467ProbePoints.csv',dtype=str,delimiter=',',usecols=(0,1))
		pLatLong = np.loadtxt(dir+'/Partition6467ProbePoints.csv',dtype=str,delimiter=',',usecols=(0,3,4))
		pAlt = np.loadtxt(dir+'/Partition6467ProbePoints.csv',dtype=str,delimiter=',',usecols=(0,5))
		pVelocity = np.loadtxt(dir+'/Partition6467ProbePoints.csv',dtype=str,delimiter=',',usecols=(0,6,7))
		pickle.dump(pTime,open('pTime.pckl','wb'))
		pickle.dump(pLatLong,open('pLatLong.pckl','wb'))
		pickle.dump(pAlt,open('pAlt.pckl','wb'))
		pickle.dump(pVelocity,open('pVelocity.pckl','wb'))
	else:
		# vars = [pTime, pLatLong, pAlt, pVelocity]
		# files = ['pTime.pckl', 'pLatLong.pckl', 'pAlt.pckl', 'pVelocity.pckl']
		pTime = pickle.load(open('pTime.pckl','rb'))
		pLatLong = pickle.load(open('pLatLong.pckl','rb'))
		pAlt = pickle.load(open('pAlt.pckl','rb'))
		pVelocity = pickle.load(open('pVelocity.pckl','rb'))
		# for i,j in zip(vars, files):
			# i = pickle.load(open(j,'rb'))
	# if not os.path.exists('lData.pckl'):
	# 	lData = np.loadtxt(dir+'/Partition6467LinkData.csv',dtype=str,delimiter=',')
	# 	pickle.dump(lData,open('lData.pckl','wb'))
	# else:
	# 	lData = pickle.load(open('lData.pckl','rb'))
	return pTime, pLatLong, pAlt, pVelocity

if __name__ == '__main__':
	# x,y = loadData('probe_data_map_matching')[:2]
	# print(x.shape)
	# print(y.shape)
	# print(x[1,1])

	ID, date_time = loadTime('probe_data_map_matching')
	# print(ID[:10])
	times = timeSlots(ID, date_time)
	# for i,x in enumerate(times.items()):
	# 	if i==10:
	# 		break
	# 	k,v = x
	# 	print('{}: {}'.format(k,v))
	
	k,v = list(times.items())[0]
	x = v[-1]-v[0]
	print(x)

	# t = date_time[10]
	# ten_min = timedelta(minutes=10)
	# print(t.isoformat(' '))
	# t = t+ten_min
	# print(t.isoformat(' '))

	graph = loadLink('probe_data_map_matching')
	for i,x in enumerate(graph.items()):
		if i==10:
			break
		k,v = x
		print('{}: {}'.format(k,v))
	print(len(graph))
	# print(graph["162844982"])
	# print(fixString("b'16244982'"))
