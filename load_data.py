import numpy as np
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt
# import time
from datetime import datetime, timezone, timedelta
import json

def loadTime(dir):
	x,y = np.loadtxt(dir+'/Partition6467ProbePoints.csv', dtype=str, delimiter=',', usecols=(0,1), unpack=True)
	x = np.array([fixString(x[i]) for i in range(x.shape[0])])
	y = np.array([datetime.strptime(y[i], "b'%m/%d/20%y %I:%M:%S %p'").timestamp() for i in range(y.shape[0])])
	return x,y

def loadProbeLatLong(dir):
	x,y = np.loadtxt(dir+'/Partition6467ProbePoints.csv', delimiter=',', usecols=(3,4), unpack=True)
	return x,y

def loadLinkLatLong(dir):
	ID,x = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,14,), unpack=True)
	x = np.array([fixString(x[i]) for i in range(x.shape[0])])
	ID = np.array([fixString(ID[i]) for i in range(ID.shape[0])])
	IDx = defaultdict(lambda: [])
	IDy = defaultdict(lambda: [])
	for i in range(x.shape[0]):
		temp = x[i].split('|')
		for comb in temp:
			comb.split('/')
			lat = float(comb[0])
			lng = float(comb[1])
			# TODO: Ignoring elvation for now
			IDx[ID[i]].append(lat)
			IDy[ID[i]].append(lng)
	return IDx, IDy

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
	# five_min = timedelta(minutes = 5)
	# uni_ids,ind,counts = np.unique(ids, return_index=True, return_counts=True)
	# times = defaultdict(lambda: [])
	# dt = date_time.copy()
	sind = date_time.argsort()
	ids1 = ids[sind]
	dt = date_time[sind]
	slots = defaultdict(lambda: defaultdict(lambda: []))
	print('Sorted wrt time\n')
	i = 0
	proc=0.
	while i < dt.shape[0]:
		time = dt[i]
		r = np.where((dt > time) & (dt <= (time+300)))
		r = r[0]
		if r.shape[0] > 1:
			uni_ids, counts = np.unique(ids1[r], return_counts=True)
			x = np.where(counts > 1)[0]
			uni_ids = uni_ids[x].tolist()
			for j in range(r.shape[0]):
				if ids[r[j]] in uni_ids:
					slots[time][ids[r[j]]].append(dt[r[j]])
			# slots[time].sort(key=lambda x: x[1])
		if r.shape[0] > 0:
			i += r.shape[0]
		else:
			i += 1
		proc = (float(i)/dt.shape[0])*100
		print('\rCompleted: {:.2f}'.format(proc),end=' ')
	print('\n')

	# for i in range(uni_ids.shape[0]):
	# 	count = 0
	# 	while count < counts[i]:
	# 		times[uni_ids[i]].append(date_time[ind[i]+count])
	# 		count += 1
	# min_datetime = None
	# for i in range(uni_ids.shape[0]):
	# 	if i == 0:
	# 		min_datetime = date_time[ind[i]]
	# 	else:
	# 		if min_datetime > date_time[ind[i]]:
	# 			min_datetime = date_time[ind[i]]
	# print('Creating Pickle dump: "slots.pckl"')
	# pickle.dump(slots,open('slots.pckl','wb'), protocol=pickle.HIGHEST_PROTOCOL)
	# print('Created Pickle Dump')
	return slots

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

	# ID, date_time = loadTime('probe_data_map_matching')
	# # print(ID[:10])
	print('Loaded Data')
	# slots = timeSlots(ID, date_time)
	slots = json.load(open('slots.pckl','r'))
	for i,x in enumerate(slots.items()):
		if i==5:
			break
		k,v = x
		print('{}: {}'.format(k,v))

	# json.dump(slots,open('slots.pckl','w'))

	# print(min.isoformat(' '))
	# k,v = list(times.items())[0]
	# x = v[-1] < v[0]
	# print(x)

	# t = date_time[10]
	# ten_min = timedelta(minutes=10)
	# print(t.isoformat(' '))
	# t = t+ten_min
	# print(t.isoformat(' '))

	# graph = loadLink('probe_data_map_matching')
	# for i,x in enumerate(graph.items()):
	# 	if i==10:
	# 		break
	# 	k,v = x
	# 	print('{}: {}'.format(k,v))
	# print(len(graph))
	# print(graph["162844982"])
	# print(fixString("b'16244982'"))

	'''Plotting'''

	# lat, lng = loadLatLong('probe_data_map_matching')
	# plt.plot(lat,lng, marker='+', color='blue')
	# plt.xlabel('Latitude')
	# plt.ylabel('Longitude')
	# plt.show()

	# IDx, IDy = loadLinkLatLong('probe_data_map_matching')
	# for j,i in enumerate(IDx):
	# 	if j==3:
	# 		break
	# 	plt.plot(IDx[i],IDy[i], marker='o', linestyle='-', c='green', mfc='red',linewidth=0.2)
	# plt.xlabel('Latitude')
	# plt.ylabel('Longitude')
	# plt.show()
