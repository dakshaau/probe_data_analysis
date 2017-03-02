from load_data import *
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict

def calculatePD(P1, P2, P3):
	x1, y1 = P1
	x2, y2 = P2
	x3, y3 = P3
	ER = 3959.
	p1 = np.array([x1,y1])
	p2 = np.array([x2,y2])
	p3 = np.array([x3,y3])
	p1_p2 = np.sum(np.square(p2-p1))
	_mu = np.sum((p3-p1)*(p2-p1))/p1_p2
	p = p1 + _mu*(p2-p1)
	pi = np.pi
	
	R = p*(pi/180.)
	R3 = p3*(pi/180.)
	ab = R3-R
	arcsin = lambda x: np.arcsin(x)
	sin = lambda x: np.sin(x)
	cos = lambda x: np.cos(x)
	PD = ER * arcsin(np.sqrt( sin(ab[0]/2.)**2 + cos(R3[0])*cos(R[0])*(sin(ab[1]/2.))**2) )
	return PD

def calculateHE(P1, P2, heading):
	arctan = lambda x: np.arctan(x)
	pi = np.pi
	delta = np.subtract(P2,P1)
	delta *= pi/180.
	# print(delta)
	theta = None
	if delta[0] < 0 and delta[1] >= 0:
		temp = -1.*pi/2.
		if delta[0] != 0 :
			temp = arctan(delta[1]/delta[0])
		theta = (2.5*pi - temp) * 180/pi
	else:
		temp = pi/2.
		if delta[0] == 0 and delta[1] < 0:
			temp = -1*temp
		if delta[0] != 0 :
			temp = arctan(delta[1]/delta[0])
		theta = (0.5*pi - temp) * 180/pi
	HE = abs(heading - theta)
	# print(theta, heading)
	if HE > 180:
		HE = 360 - HE
	return HE

def createCandidate(P, l_x, l_y, p_speed, p_head):
	candidates = []
	PDs = []
	# HEs = []
	for id in l_x:
		for i,c in enumerate(zip(l_x[id], l_y[id])):
			if i == len(l_x[id])-1:
				break
			x1,y1 = c
			x2 = l_x[id][i+1]
			y2 = l_x[id][i+1]
			pd = calculatePD((x1,y1), (x2,y2), P)
			# PDs.append([id,x1,y1,pd])
			he = calculateHE((x1,y1), (x2,y2), p_head)
			PDs.append([id,x1,y1,he,pd])
	PDs = np.asanyarray(PDs,)
	sind = PDs.argsort(axis=0)
	PDs = PDs[sind[:,4],:]
	# print(PDs.dtype)
	if p_speed < 7.:
		candidates = PDs[:4,:3].tolist()
	else:
		arr = np.asarray(PDs[:,3],dtype = np.float32)
		# print(arr.shape)
		# print(arr.dtype)
		ind = np.where(arr <= 45.0)
		ind = ind[0]
		PDs = PDs[ind,:]
		candidates = PDs[:2,:3].tolist()
	return candidates

def TTP(slot_data, l_x, l_y, p_speed, p_head):
	p_x, p_y = slot_data
	# print(p_x)
	if p_x.shape[0] <2:
		# print('Not enough points ...')
		return {}
	candidates = defaultdict(lambda: [])
	for i in range(p_x.shape[0]):
		x = p_x[i]
		y = p_y[i]
		candidates[str(x)+','+str(y)] = createCandidate((x,y), l_x, l_y, p_speed[i], p_head[i])
	return candidates

def MapMatching(p_id, d_t, p_x, p_y, slots, l_x, l_y, p_speed, p_head):
	x = None
	cand = defaultdict(lambda: {})
	print('Creating Candidates ... ')
	prog = 0.
	print('Completed: {:.2f}'.format(prog),end=' ')
	for j,k in enumerate(slots):
		# print(slots[k])
		tot = len(slots[k])
		for j,i in enumerate(slots[k]):
			ind = np.where((p_id == i) & ((d_t >= slots[k][i][0]) & (d_t <= slots[k][i][-1])))[0]
			x = TTP((p_x[ind], p_y[ind]), l_x, l_y, p_speed[ind], p_head[ind])
			cand[k][i] = x
		# break
		prog = (j/(float(tot)-1.)) * 100
		print('\rCompleted: {:.2f}'.format(prog),end = ' ') 
	print()
	# print(cand)
	json.dump(cand,open('candidates.json','w'))
	# print(x)

if __name__ == '__main__':
	dat = 'probe_data_map_matching'
	p_id, d_t = loadTime(dat)
	slots = timeSlots(p_id, d_t)
	p_x, p_y = loadProbeLatLong(dat)
	l_x, l_y = loadLinkLatLong(dat)
	p_speed = loadProbeSpeed(dat)
	p_head = loadProbeHeading(dat)
	
	MapMatching(p_id, d_t, p_x, p_y, slots, l_x, l_y, p_speed, p_head)

	# print(p_speed[:10])
	# print(p_head[:10])
	# for j,x3 in enumerate(p_x):
	# 	if j==1:
	# 		break
	# for j,x1 in enumerate(l_x):
	# 	if j==1:
	# 		break

	# x3 = p_x[0]
	# y3 = p_y[0]

	# x = l_x[x1]
	# y = l_y[x1]

	# print(calculateHE((x[0], y[0]),(x[1], y[1]),p_head[0]))
