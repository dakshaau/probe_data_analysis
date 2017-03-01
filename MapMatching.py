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

def createCandidate():
	pass

if __name__ == '__main__':
	dat = 'probe_data_map_matching'
	# p_id, d_t = loadTime(dat)
	# slots = timeSlots(p_id, d_t)
	p_x, p_y = loadProbeLatLong(dat)
	l_x, l_y = loadLinkLatLong(dat)
	# p_speed = loadProbeSpeed(dat)
	# p_head = loadProbeHeading(dat)
	# print(p_speed[:10])
	# print(p_head[:10])
	for j,x3 in enumerate(p_x):
		if j==1:
			break
	for j,x1 in enumerate(l_x):
		if j==1:
			break

	x3 = p_x[0]
	y3 = p_y[0]

	x = l_x[x1]
	y = l_y[x1]


	print(calculatePD((x[0], y[0]),(x[1], y[1]),(x3,y3)))
