from load_data import *
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
# import threading
from multiprocessing import Process

def calculateTheta(P1, P2):
	arctan = lambda x,y: np.arctan2(x,y)
	pi = np.pi
	delta = P2-P1
	theta = np.zeros((P1.shape[0],),dtype=P1.dtype)
	# ind = np.where(np.isnan(delta))
	# print(delta[ind])
	ind_t = np.where((delta[:,0] < 0) & (delta[:,1] >= 0))[0]
	ind_f = np.where(~((delta[:,0] < 0) & (delta[:,1] >= 0)))[0]
	# ind_f = np.where((~((delta[:,0] < 0) & (delta[:,1] >= 0))) & (delta[:,0] != 0))[0]
	theta[ind_t] = (2.5*pi - arctan(delta[ind_t,1],delta[ind_t,0])) * 180./pi
	theta[ind_f] = (0.5*pi - arctan(delta[ind_f,1],delta[ind_f,0])) * 180./pi
	
	# ind = np.where(np.isnan(theta))[0]
	# # print(ind)

	# ind_neg = np.where((delta[ind_f,0] == 0) & (delta[ind_f,1] < 0))[0]
	# ind_pos = np.where((delta[ind_f,0] == 0) & (delta[ind_f,1] >= 0))[0]

	# theta[ind_neg] = (0.5*pi + pi/2.) * 180./pi
	# theta[ind_pos] = (0.5*pi - pi/2.) * 180./pi

	# ind = np.where(theta > 360)[0]
	# while ind.shape[0] > 0:
	# 	theta[ind] -= 360
	# 	ind = np.where(theta > 360)[0]

	return theta

def calculatePD(P1, P2, P3):
	# x1, y1 = P1
	# x2, y2 = P2
	x3, y3 = P3
	ER = 3956.
	# p1 = np.array([x1,y1])
	# p2 = np.array([x2,y2])
	p3 = np.array([x3,y3])
	p1_p2 = np.sum(np.square(P2-P1),axis=1)
	x = np.sum((p3-P1)*(P2-P1),axis=1)
	ind = np.where(p1_p2 != 0)[0]
	_mu = np.zeros(p1_p2.shape,dtype=p1_p2.dtype)
	_mu[ind] = x[ind]/p1_p2[ind]
	p = P1 + np.vstack((_mu,_mu)).T*(P2-P1)
	pi = np.pi
	
	R = p*(pi/180.)
	R3 = p3*(pi/180.)
	ab = R3-R
	arcsin = lambda x: np.arcsin(x)
	sin = lambda x: np.sin(x)
	cos = lambda x: np.cos(x)
	PD = ER * arcsin(np.sqrt( sin(ab[:,0]/2.)**2 + cos(R3[0])*cos(R[:,0])*(sin(ab[:,1]/2.))**2) )
	return PD

def calculateHE(theta, heading):
	HE = np.absolute(heading - theta)
	ind_great = np.where(HE > 180)[0]
	HE[ind_great] = 360. - HE[ind_great]
	# HE = np.absolute(HE)
	return HE

def createCandidate(P, l_id, P1, P2, p_speed, p_head, theta):
	PDs = calculatePD(P1, P2, P)
	HEs = calculateHE(theta, p_head)
	sind = PDs.argsort()
	PDs = PDs[sind]
	HEs = HEs[sind]
	IDs = l_id[sind]
	p1 = P1[sind,:]

	if p_speed < 7.:
	# if True:
		if IDs.shape[0] >= 4:
			return [(IDs[i], p1[i,0], p1[i,1]) for i in range(4)]
		else:
			return [(IDs[i], p1[i,0], p1[i,1]) for i in range(IDs.shape[0])]
	else:
		# print(HEs)
		ind = np.where(HEs <= 90)[0]
		if ind.shape[0] > 0:
			PDs = PDs[ind]
			IDs = IDs[ind]
			p1 = p1[ind,:]
		# print(p1.shape)
			if ind.shape[0] >= 4:
				return [(IDs[i], p1[i,0], p1[i,1]) for i in range(4)]
			else:
				return [(IDs[i], p1[i,0], p1[i,1]) for i in range(ind.shape[0])]
		else:
			return []

def TTP(slot_data, l_id, P1, P2, p_speed, p_head, theta):
	p_x, p_y = slot_data
	# print(p_x)
	# ER = 6371.
	# dLat = (0.1/ER)*(180/np.pi)
	dLat = 0.0001
	dLong = 0.0001
	if p_x.shape[0] <2:
		# print('Not enough points ...')
		return {}
	candidates = defaultdict(lambda: [])
	for i in range(p_x.shape[0]):
		x = p_x[i]
		y = p_y[i]
		'''
		Creating Pseudo Link Set using error rectangle with sides of length 20 meters.
		'''

		# dLong = (0.1/ER)*(180/np.pi)/np.cos(x*np.pi/180)
		rxMin = x - dLat
		rxMax = x + dLat
		ryMin = y - dLong
		ryMax = y + dLong
		ind = np.where(((P1[:,0] >= rxMin) & (P1[:,0] <= rxMax)) & ((P1[:,1] >= ryMin) & (P1[:,1] <= ryMax)))[0]
		if ind.shape[0] == 0:
			candidates[str(x)+','+str(y)] = []
			continue
		# print(ind.shape)
		p1_f = P1[ind,:]
		p2_f = P2[ind,:]
		lid_f = l_id[ind]
		theta_f = theta[ind]
		candidates[str(x)+','+str(y)] = createCandidate((x,y), lid_f, p1_f, p2_f, p_speed[i], p_head[i], theta_f)
	return candidates

def MapMatching(p_id, d_t, p_x, p_y, slots, l_id, P1, P2, p_speed, p_head, theta, Pname = 'Main'):
	x = None
	# cand = defaultdict(lambda: {})
	prog = 0.
	# print('Completed: {:.2f}'.format(prog),end=' ')
	tot = len(slots)
	
	for j,k in enumerate(slots):
		# print(slots[k])
		if os.path.exists('slot_cand/{}.json'.format(k)):
			prog = (j/(float(tot)-1.)) * 100
			print('\rCompleted : {:.2f}%, Process: {}'.format(prog, Pname),end=' ')
			continue
		cand = {}
		# tot = len(slots[k])
		# print('Completed : {:.2f}%'.format(prog),end=' ')
		for y,i in enumerate(slots[k]):
			ind = np.where((p_id == i) & ((d_t >= slots[k][i][0]) & (d_t <= slots[k][i][-1])))[0]
			x = TTP((p_x[ind], p_y[ind]), l_id, P1, P2, p_speed[ind], p_head[ind], theta)
			cand[i] = x
			# prog = (y/(float(tot)-1.)) * 100
			# print('\rCompleted : {:.2f}%'.format(prog),end=' ')	
		# break
		prog = (j/(float(tot)-1.)) * 100
		# print('Creating {}.json'.format(k)) 
		print('\rCompleted : {:.2f}%, Process: {}'.format(prog, Pname),end=' ')
		json.dump(cand,open('slot_cand/{}.json'.format(k),'w'))
		del cand

	# print()
	# print(cand)
	# print(x)

# class MMThread(threading.Thread):
# 	slots = None
# 	p_id=d_t=p_x=p_y=slots=l_id=P1=P2=p_speed=p_head=theta= None

# 	def __init__(self, p_id, d_t, p_x, p_y, slots, l_id, P1, P2, p_speed, p_head, theta):
# 		threading.Thread.__init__(self)
# 		self.slots = slots
# 		self.p_id = p_id
# 		self.d_t = d_t
# 		self.p_x = p_x
# 		self.p_y = p_y
# 		self.P1 = P1
# 		self.P2 = P2
# 		self.p_speed = p_speed
# 		self.p_head = p_head
# 		self.l_id = l_id
# 		self.theta = theta

# 	def run(self):
# 		MapMatching(self.p_id, self.d_t, self.p_x, self.p_y, self.slots, self.l_id, self.P1, self. P2, self.p_speed, self.p_head, self.theta)

if __name__ == '__main__':
	dat = 'probe_data_map_matching'
	p_id, d_t = loadTime(dat)
	slots = timeSlots(p_id, d_t)
	p_x, p_y = loadProbeLatLong(dat)
	l_id, l_x, l_y = loadLinkLatLong(dat)
	p_speed = loadProbeSpeed(dat)
	p_head = loadProbeHeading(dat)
	# linkGraph = loadLink(dat)[1]
	dot = loadLinkDOT(dat)
	l_id, P1, P2 = createP1P2(l_id, l_x, l_y, dot)

	# l_id, l_x, l_y = getLinkXYArray(l_x, l_y)

	# l_id = l_id[:-1]

	# P = np.vstack((l_x,l_y))
	# P = P.T

	# P1 = P[:-1]
	# l1 = l_id[:-1]
	# P2 = P[1:]
	# l2 = l_id[1:]

	# ind = np.where(l1 == l2)
	# l_id = l1[ind]
	# P1 = P1[ind]
	# P2 = P2[ind]


	theta = calculateTheta(P1,P2)

	# print(np.where(theta < 0)[0].shape)
	# print(np.where(theta > 360)[0].shape)
	# print(np.where(p_head < 0)[0].shape)
	# print(np.where(p_head > 360)[0].shape)

	# HE = calculateHE(theta, 45.)

	# print(HE[:10])
	
	# pd = calculatePD(P1, P2, (51.60, 8.90))

	# print(pd[:10])

	# x = np.isnan(theta)

	# x = np.where(x)

	# print(np.min(theta))

	# print(np.where(theta == np.nan))
	# print('{}, {}'.format(l_id.shape, l_id.dtype))
	# print('{}, {}'.format(P1.shape, P1.dtype))
	# print('{}, {}'.format(l_x.T.shape, l_x.dtype))
	# print('{}, {}'.format(l_y.T.shape, l_y.dtype))
	# print(P1[:10,1]==l_y[:10])
	# print

	# print(l_id[:10])
	# MapMatching(p_id, d_t, p_x, p_y, slots, l_id, P1, P2, p_speed, p_head, theta)

	''' Multiprocessing : Creating 4 Processes '''
	x = len(slots)       ## This is just for dividing the data between multiple systems
	part = int(x/4)
	slots = OrderedDict(sorted(list(slots.items()), key=lambda x: x[0]))

	x = len(slots)
	print('Total number of slots: {}'.format(x))
	part = int(x/4)
	t1 = Process(target = MapMatching, args=(p_id, d_t, p_x, p_y, OrderedDict(list(slots.items())[:part]), l_id, P1, P2, p_speed, p_head, theta, 'P1'))
	t2 = Process(target = MapMatching, args=(p_id, d_t, p_x, p_y, OrderedDict(list(slots.items())[part : 2*part]), l_id, P1, P2, p_speed, p_head, theta, 'P2'))
	t3 = Process(target = MapMatching, args=(p_id, d_t, p_x, p_y, OrderedDict(list(slots.items())[2*part : 3*part]), l_id, P1, P2, p_speed, p_head, theta, 'P3'))
	t4 = Process(target = MapMatching, args=(p_id, d_t, p_x, p_y, OrderedDict(list(slots.items())[3*part:]), l_id, P1, P2, p_speed, p_head, theta, 'P4'))
	
	t1.start()
	t2.start()
	t3.start()
	t4.start()

	t1.join()
	# print('{} slots done ...'.format(x/4))
	t2.join()
	# print('{} slots done ...'.format(2*x/4))
	t3.join()
	# print('{} slots done ...'.format(3*x/4))
	t4.join()
	# print('{} slots done ...'.format(x))

	

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
