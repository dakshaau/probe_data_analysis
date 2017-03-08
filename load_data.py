import numpy as np
import pickle
import os
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
# import time
from datetime import datetime, timezone, timedelta
import json

def loadTime(dir):
	x=y= None
	if not os.path.exists('probeID.pckl') or not os.path.exists('probeTime.pckl'):
		x,y = np.loadtxt(dir+'/Partition6467ProbePoints.csv', dtype=str, delimiter=',', usecols=(0,1), unpack=True)
		x = np.array([fixString(x[i]) for i in range(x.shape[0])])
		y = np.array([datetime.strptime(y[i], "b'%m/%d/20%y %I:%M:%S %p'").timestamp() for i in range(y.shape[0])])
		pickle.dump(x,open('probeID.pckl','wb'))
		pickle.dump(y,open('probeTime.pckl','wb'))
		print('Created dumps ...')
	else:
		print('Loading Data ...')
		x = pickle.load(open('probeID.pckl','rb'))
		y = pickle.load(open('probeTime.pckl','rb'))
	return x,y

def loadProbeSpeed(dir):
	if os.path.exists('probeSpeed.pckl'):
		speed = pickle.load(open('probeSpeed.pckl','rb'))
		return speed
	speed = np.loadtxt(dir+'/Partition6467ProbePoints.csv', delimiter=',', usecols=(6,))
	speed = 0.621371 * speed
	pickle.dump(speed, open('probeSpeed.pckl','wb'))
	return speed

def loadProbeHeading(dir):
	if os.path.exists('probeHeading.pckl'):
		heading = pickle.load(open('probeHeading.pckl','rb'))
		return heading
	heading = np.loadtxt(dir+'/Partition6467ProbePoints.csv', delimiter=',', usecols=(7,))
	pickle.dump(heading, open('probeHeading.pckl','wb'))
	return heading

def loadProbeLatLong(dir):
	x=y= None
	if not os.path.exists('probeX.pckl') or not os.path.exists('probeY.pckl'):
		x,y = np.loadtxt(dir+'/Partition6467ProbePoints.csv', delimiter=',', usecols=(3,4), unpack=True)
		pickle.dump(x,open('probeX.pckl','wb'))
		pickle.dump(y,open('probeY.pckl','wb'))
	else:
		x = pickle.load(open('probeX.pckl','rb'))
		y = pickle.load(open('probeY.pckl','rb'))
	return x,y

def loadLinkLatLong(dir):
	if os.path.exists('linkX.pckl') and os.path.exists('linkY.pckl') and os.path.exists('linkID.pckl'):
		ID = pickle.load(open('linkID.pckl','rb'))
		X = pickle.load(open('linkX.pckl','rb'))
		Y = pickle.load(open('linkY.pckl','rb'))
		return ID, X, Y
	ID,x = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,14,), unpack=True)
	x = np.array([fixString(x[i]) for i in range(x.shape[0])])
	ID = np.array([fixString(ID[i]) for i in range(ID.shape[0])])
	IDx = defaultdict(lambda: [])
	IDy = defaultdict(lambda: [])
	for i in range(x.shape[0]):
		temp = x[i].split('|')
		for comb in temp:
			comb = comb.split('/')
			lat = float(comb[0])
			lng = float(comb[1])
			# TODO: Ignoring elvation for now
			IDx[ID[i]].append(lat)
			IDy[ID[i]].append(lng)
	del ID
	ID, X, Y = getLinkXYArray(IDx, IDy)
	# json.dump(IDx, open('linkX.json','w'))
	# json.dump(IDy, open('linkY.json','w'))
	pickle.dump(ID, open('linkID.pckl','wb'))
	pickle.dump(X, open('linkX.pckl','wb'))
	pickle.dump(Y, open('linkY.pckl','wb'))
	return ID, X, Y

def getLinkXYArray(X, Y):
	IDs = []
	Xs = []
	Ys = []
	for k in X:
		for x,y in zip(X[k],Y[k]):
			IDs.append(k)
			Xs.append(x)
			Ys.append(y)
	IDs = np.asarray(IDs)
	Xs = np.asarray(Xs, dtype=np.float64)
	Ys = np.asarray(Ys, dtype=np.float64)
	return IDs, Xs, Ys

def fixString(x):
	return x.split("'")[-2]

def loadLink(dir):
	x,y,z,cat = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,1,2,5), unpack=True)
	dist = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=float, delimiter=',', usecols=(3,))
	graph = defaultdict(lambda: [])
	lengths = defaultdict(lambda: [])
	for i in range(y.shape[0]):
		Y = fixString(y[i])
		Z = fixString(z[i])
		C = fixString(cat[i])
		if C == 'F':
			graph[Y].append(Z)
			lengths[Y].append((Z,dist[i]))
		elif C == 'T':
			graph[Z].append(Y)
			lengths[Z].append((Y,dist[i]))
		elif C == 'B':
			graph[Y].append(Z)
			lengths[Y].append((Z,dist[i]))
			graph[Z].append(Y)
			lengths[Z].append((Y,dist[i]))
		# graph[y[i]].append(z[i])

	return graph, lengths

def timeSlots(ids, date_time):
	slots = {}
	if not os.path.exists('slots.json'):
		sind = date_time.argsort()
		ids1 = ids[sind]
		dt = date_time[sind]
		slots = defaultdict(lambda: defaultdict(lambda: []))
		print('Sorted wrt time\n')
		i = 0
		proc=0.
		# counter = 0
		print('Making SLots')
		while i < dt.shape[0]:
			time = dt[i]
			r = np.where((dt >= time) & (dt < (time+600)))
			r = r[0]
			if r.shape[0] > 1:
				uni_ids, counts = np.unique(ids1[r], return_counts=True)
				x = np.where(counts > 1)[0]
				uni_ids = uni_ids[x].tolist()
				for j in range(r.shape[0]):
					if ids1[r[j]] in uni_ids:
						slots[time][ids1[r[j]]].append(dt[r[j]])
			if r.shape[0] > 0:
				i += r.shape[0]
			else:
				i += 1
			proc = (float(i)/dt.shape[0])*100
			print('\rCompleted: {:.2f}'.format(proc),end=' ')
		print('\n')
		# slots = dict(sorted(slots.items(),key=lambda x: x[0]))
		slots = OrderedDict(dict(slots))
		print('Creating JSON: "slots.json"')
		json.dump(slots,open('slots.json','w'))
	else:
		print('Loading "slots.json" ...')
		slots = json.load(open('slots.json','r'), object_pairs_hook=OrderedDict)
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
	# # # # print(ID[:10])
	# print('Loaded Data: {} points'.format(ID.shape[0]))
	# # print(ID[:200])
	# # print()
	# # print(date_time[:200])
	# slots = timeSlots(ID, date_time)

	# # # slots = json.load(open('slots.pckl','r'))
	# count = 0
	# for i,x in enumerate(slots.items()):
	# 	# if i==5:
	# 	# 	break
	# 	k,v = x
	# 	# print('{}: {}\n'.format(k,v))
	# 	for y in v:
	# 		count += len(v[y])
	# print(count)

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

	a, b, c = loadLinkLatLong('probe_data_map_matching')
	# for j,i in enumerate(IDx):
	# 	# if j==3:
	# 	# 	break
	# 	# k,v = i
	# 	# print('{}: {}'.format(k,v))
	# 	plt.plot(IDx[i],IDy[i],marker='o', linestyle='-', c='green', mfc='red',linewidth=2)
	# plt.xlabel('Latitude')
	# plt.ylabel('Longitude')
	# plt.show()
