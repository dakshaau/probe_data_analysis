from load_data import *
import os
import json
import pickle
from collections import defaultdict
from datetime import datetime
import itertools as it
from matplotlib import pyplot as plt
import numpy as np

def generatePairs(p1, p2):
	if len(p1) == 0 or len(p2) == 0:
		return []
	else:
		pairs = []
		for i in p1:
			for j in p2:
				pairs.append((i,j))
		return pairs

def isChild(node1, node2, graph, visited, result, level):
	if not visited[node1]:
		visited[node1] = True
		children = graph[node1]
		if level > 4:
			result[node1] = False
			return False
		if len(children) == 0:
			result[node1] = False
			return False
		if node2 in children:
			result[node1] = True
			return True
		else:
			for i in children:
				if isChild(i,node2,graph,visited,result, level+1):
					result[node1] = True
					return True
			result[node1] = False
			return False
	else:
		return result[node1]

def isConnected(id1, id2, graph, lids, dot):
	if id1 == id2:
		return True
	else:
		# print(lids[id1])
		ref1, nref1 = lids[id1][0]
		ref2, nref2 = lids[id2][0]
		visited = defaultdict(lambda: False)
		result = defaultdict(lambda: False)
		if dot[id1] == 'F' and dot[id2] =='T':
			return isChild(nref1, nref2, graph, visited, result, 0)
		elif dot[id1] == 'T' and dot[id2] =='F':
			return isChild(ref1, ref2, graph, visited, result, 0)			
		elif dot[id1] == 'F' and dot[id2] == 'F':
			return isChild(nref1, ref2, graph, visited, result, 0)
		elif dot[id1] == 'T' and dot[id2] == 'T':
			return isChild(ref1, nref2, graph, visited, result, 0)
		elif dot[id1] == 'B' and dot[id2] != 'B':
			if dot[id2] == 'T':
				a = isChild(ref1, nref2, graph, visited, result, 0)
				b = isChild(nref1, nref2, graph, visited, result, 0)
				return a or b
			elif dot[id2] == 'F':
				a = isChild(ref1, ref2, graph, visited, result, 0)
				b = isChild(nref1, ref2, graph, visited, result, 0)
				return a or b
		elif dot[id2] == 'B' and dot[id1] != 'B':
			if dot[id1] == 'T':
				a = isChild(ref1, ref2, graph, visited, result, 0)
				b = isChild(ref1, nref2, graph, visited, result, 0)
				return a or b
			elif dot[id1] == 'F':
				a = isChild(nref1, ref2, graph, visited, result, 0)
				b = isChild(nref1, nref2, graph, visited, result, 0)
				return a or b
		elif dot[id2] == 'B' and dot[id1] == 'B':
			a = isChild(ref1, nref2, graph, visited, result, 0)
			b = isChild(nref1, ref2, graph, visited, result, 0)
			c = isChild(nref1, nref2, graph, visited, result, 0)
			d = isChild(ref1, ref2, graph, visited, result, 0)
			return a or b or c or d

def createRoutes(routes):
	x = len(routes)
	n = 0
	pr = {}
	while n < x:
		# print(len(pr))
		l = len(routes[n])
		if len(pr) == 0:
			if l != 0:
				# t = tuple()
				# [pr.append([i]) for i in routes[n]]
				# for i in routes[n]:
				# 	t = t+tuple((i,))
				# pr.append(t)
				pr[0] = [i for i in routes[n]]
			else:
				pr[0] = []
				# pr.append(())
		else:
			if l == 0:
				n += 1
				continue
			elif l > 1:
				# [pr.append() for i in range(l-1)]
				temp = sorted(routes[n]*len(pr))
				# pr = list(pr.items())*l
				# tot = len(pr) * l
				k = list(pr.keys())
				# print(len(k))
				for ix in k:
					for j in range(l-1):
						pr[ix+len(k)*(j+1)] = pr[ix][:]
						# print(ix, ix+len(k)*(j+1))

				for i in range(len(pr)):
					pr[i].append(temp[i])
				del temp[:]
			elif l == 1:
				for i in range(len(pr)):
					pr[i].append(routes[n][0])
		n += 1
	return pr

def calculateTheta(P1, P2):
	arctan = lambda x,y: np.arctan2(x,y)
	pi = np.pi
	delta = P2-P1
	theta = np.zeros((P1.shape[0],),dtype=P1.dtype)
	ind_t = np.where((delta[:,0] < 0) & (delta[:,1] >= 0))[0]
	ind_f = np.where(~((delta[:,0] < 0) & (delta[:,1] >= 0)))[0]
	theta[ind_t] = (2.5*pi - arctan(delta[ind_t,1],delta[ind_t,0])) * 180./pi
	theta[ind_f] = (0.5*pi - arctan(delta[ind_f,1],delta[ind_f,0])) * 180./pi
	
	return theta

def calculatePD(P1, P2, P3):
	# x1, y1 = P1
	# x2, y2 = P2
	x3, y3 = P3
	ER = 6371.
	# p1 = np.array([x1,y1])
	# p2 = np.array([x2,y2])
	p3 = np.array([x3,y3])
	
	p1_p2 = np.sum(np.square(P2-P1),axis=1)
	x = np.sum((p3-P1)*(P2-P1),axis=1)
	ind = np.where(p1_p2 != 0)[0]
	_mu = np.zeros(p1_p2.shape,dtype=p1_p2.dtype)
	_mu[ind] = x[ind]/p1_p2[ind]
	p = P1 + np.vstack((_mu,_mu)).T*(P2-P1)
	# print(p[0])
	pi = np.pi
	
	R = p*(pi/180.)
	R3 = p3*(pi/180.)
	ab = R3-R
	arcsin = lambda x: np.arcsin(x)
	sin = lambda x: np.sin(x)
	cos = lambda x: np.cos(x)
	PD = 2 * ER * arcsin(np.sqrt( sin(ab[:,0]/2.)**2 + cos(R3[0])*cos(R[:,0])*(sin(ab[:,1]/2.)**2)) )
	return PD, p

def haversineDist(P1, P2):
	# x,y = P2
	# p3 = np.array([x,y])
	ER = 6371.
	pi = np.pi
	p = P1.copy()
	p3 = P2.copy()
	R = p*(pi/180.)
	R3 = p3*(pi/180.)
	ab = R3-R
	arcsin = lambda x: np.arcsin(x)
	sin = lambda x: np.sin(x)
	cos = lambda x: np.cos(x)
	PD = 2 * ER * arcsin(np.sqrt( sin(ab[0]/2.)**2 + cos(R3[0])*cos(R[0])*(sin(ab[1]/2.)**2)) )
	return PD

def calculateHE(theta, heading):
	HE = np.absolute(heading - theta)
	ind_great = np.where(HE > 180)[0]
	HE[ind_great] = 360. - HE[ind_great]
	# HE = np.absolute(HE)
	return HE

def createRoute(routes, graph, lids, dot, coors):
	x = len(routes)
	n = 0
	res = [[]]
	resc = [[]]
	nr = 0
	while n < x-1:
		# if n == 0:
		for i, li in enumerate(routes[n]):
			if len(routes[n+1]) == 1:
				if li[1] == routes[n+1][0]:
					res[nr].append(li[0])
					res[nr].append(li[1])
					resc[nr].append((coors[n],coors[n+1]))
					break
			elif len(routes[n+1]) > 1:
				for li2 in routes[n+1]:
					if li[1] == li2[0]:
						res[nr].append(li[0])
						res[nr].append(li[1])
						resc[nr].append((coors[n],coors[n+1]))
						break
			elif len(routes[n+1]) == 0:
				res[nr].append(li[0])
				res[nr].append(li[1])
				resc[nr].append((coors[n],coors[n+1]))
				nr+=1
				res.append([])
				resc.append([])
				break
			if i == len(routes[n])-1 and len(res[nr]) == 0:
				res[nr].append(li[0])
				res[nr].append(li[1])
				resc[nr].append((coors[n],coors[n+1]))
				nr+=1
				res.append([])
				resc.append([])
		# else:

		n+=1
	return res, resc

def createMapMatch(dat, fls, slots, p_id, p_x, p_y, d_t, p_speed, p_head, p_alt, l_id, P1, P2, dots, theta, lslopes):
	if os.path.exists(dat+'/MapMatchResult.csv'):
		return
	fil = open(dat+'/MapMatchResult.csv','w')
	tot = float(len(fls))
	prog = 0.
	for lx,fl in enumerate(fls):
		slot = json.load(open('CandidateLinks/{}.json'.format(fl) ,'r'))
		slot_ID = fl
		timeslot = slots[slot_ID]
		# print(slot_ID)
		for i, car in enumerate(slot):
			# if car != '201851':
			# 	continue
			# print(car)
			times = sorted(timeslot[car])
			# print(times == timeslot[car])
			# continue

			# print(times)
			# print(len(times) - len(set(times)))
			coors = slot[car][1]
			links = slot[car][0]
			# print(np.where(p_id == car)[0].shape)
			ind = np.where((p_id == car) & ((d_t >= times[0]) & (d_t <= times[-1])))[0]
			x = p_x[ind]
			y = p_y[ind]
			times = d_t[ind].tolist()
			# print(x.shape[0])
			speed = p_speed[ind]
			head = p_head[ind]
			alt = p_alt[ind]
			cc = ['{},{}'.format(str(x[k]),str(y[k])) for k in range(x.shape[0])]
			t = []
			c = []
			sp = []
			he = []
			at = []
			[(t.append(m),c.append(l), sp.append(n), he.append(o), at.append(p)) for l,m,n,o,p in zip(cc,times,speed, head, alt) if l not in c]
			# print(np.array(coors).shape)
			# print(np.array(c).shape)
			# ind = np.where(np.array(coors) == np.array(c))[0]
			ne_t = [m for l,m in zip(c,t) if l in coors]
			ne_sp = [m for l,m in zip(c,sp) if l in coors]
			ne_he = [m for l,m in zip(c,he) if l in coors]
			ne_at = [m for l,m in zip(c,at) if l in coors]
			# for x in range(ind.shape[0]):
			# 	ne_t = t[ind[x]]
			# print(len(ne_t) == len(coors))
			point_cand = defaultdict(lambda: [])
			for ix,lids in enumerate(links):
				# point_cand[coors[ix]]
				# point_cand[coors[ix+1]]
				# print(lids)
				for iy in lids:
					lid1 = iy[0]
					lid2 = iy[1]
					point_cand[coors[ix]].append(lid1)
					point_cand[coors[ix+1]].append(lid2)
				point_cand[coors[ix]] = list(set(point_cand[coors[ix]]))
				point_cand[coors[ix+1]] = list(set(point_cand[coors[ix+1]]))
			# print(len(point_cand) == len(coors))
			for ix,k in enumerate(coors):
				# if ix == 4:
					# break
				x3, y3 = map(float, k.split(','))
				p1 = []
				p2 = []
				lids = []
				indices = None
				for ids in point_cand[k]:
					ind = np.where(l_id == ids)[0]
					# [lids.append(ids) for i in range(ind.shape[0])]
					if indices is None:
						indices = ind
					else:
						indices = np.hstack((indices,ind))
					# p1.append(P1[ind].tolist()[0])
					# p2.append(P2[ind].tolist()[0])
					# lids.append(l_id[ind].tolist()[0])
				p1 = P1[indices,:]
				p2 = P2[indices,:]
				lids = l_id[indices]
				d = dots[indices]
				t = theta[indices]
				rxMax = x3 + 0.001
				rxMin = x3 - 0.001
				ryMax = y3 + 0.001
				ryMin = y3 - 0.001
				ind = np.where(((p1[:,0] >= rxMin) & (p1[:,0] <= rxMax)) & ((p1[:,1] >= ryMin) & (p1[:,1] <= ryMax)))[0]
				if ind.shape[0] == 0:
					continue
				p1 = p1[ind,:]
				p2 = p2[ind,:]
				lids = lids[ind]
				d = d[ind]
				t = t[ind]
				PD, p = calculatePD(p1, p2, (x3, y3))
				HEs = calculateHE(t, ne_he[ix])
				sind = np.argsort(PD)
				p1 = p1[sind,:]
				p2 = p2[sind,:]
				PD = PD[sind]
				HEs = HEs[sind]
				p = p[sind]
				lids = lids[sind]
				d = d[sind]
				if ne_sp[ix] < 7:
					if PD[0]*1000 > 10.:
						continue

					dist = 0.
					indexes = np.where((l_id == lids[0]) & (dots == d[0]))[0]
					p1x = P1[indexes,:]
					p2x = P2[indexes,:]
					t_id = np.where(((p1x[:,0] == p1[0,0]) & (p1x[:,1] == p1[0,1])) & ((p2x[:,0] == p2[0,0]) & (p2x[:,1] == p2[0,1])))[0]

					indexes = list(range(indexes.shape[0]))
					if d[0] == 'F':
						# px,py = p1[0,:]
						if lids[0] in lslopes:
							dists = lslopes[lids[0]]
							dist = dists[indexes[t_id[0]]][0] + (haversineDist(p1[0,:],p[0,:]) * 1000)
						else:
							dist = haversineDist(p1x[0,:],p[0,:]) * 1000
					elif d[0] == 'T':
						if lids[0] in lslopes.keys():
							dists = lslopes[lids[0]]
							indexes.reverse()
							dist = dists[indexes[t_id[0]]][0] + (haversineDist(p2[0,:],p[0,:]) * 1000)
						else:
							dist = haversineDist(p2x[-1,:],p[0,:]) * 1000
					fil.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(car, ne_t[ix], x3, y3, ne_at[ix], ne_sp[ix], ne_he[ix], lids[0], d[0], dist, PD[0]*1000))

				else:
					ind =  np.where(HEs <= 90)[0]
					if ind.shape[0] > 0:
						p1 = p1[ind,:]
						p2 = p2[ind,:]
						PD = PD[ind]
						HEs = HEs[ind]
						p = p[ind]
						lids = lids[ind]
						d = d[ind]
						if PD[0]*1000 > 10.:
							continue
						# t_id = np.where((l_id == lids[0]) & ((P1[:,0] == p1[0,0]) & (P1[:,1] == p1[0,1])) & ((P2[:,0] == p2[0,0]) & (P2[:,1] == p2[0,1])))[0]
						# direc = dots[t_id]
						dist = 0.
						indexes = np.where((l_id == lids[0]) & (dots == d[0]))[0]
						# print(indexes.shape)
						p1x = P1[indexes,:]
						p2x = P2[indexes,:]
						t_id = np.where(((p1x[:,0] == p1[0,0]) & (p1x[:,1] == p1[0,1])) & ((p2x[:,0] == p2[0,0]) & (p2x[:,1] == p2[0,1])))[0]
						indexes = list(range(indexes.shape[0]))
						if d[0] == 'F':
							# px,py = p1[0,:]
							if lids[0] in lslopes:
								dists = lslopes[lids[0]]
								dist = dists[indexes[t_id[0]]][0] + (haversineDist(p1[0,:],p[0,:]) * 1000)
							else:
								dist = haversineDist(p1x[0,:],p[0,:]) * 1000
						elif d[0] == 'T':
							if lids[0] in lslopes.keys():
								dists = lslopes[lids[0]]
								indexes.reverse()
								dist = dists[indexes[t_id[0]]][0] + (haversineDist(p2[0,:],p[0,:]) * 1000)
							else:
								dist = haversineDist(p2x[-1,:],p[0,:]) * 1000
						fil.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(car, ne_t[ix], x3, y3, ne_at[ix], ne_sp[ix], ne_he[ix], lids[0], d[0], dist, PD[0]*1000))
		prog = (lx/(tot-1)) * 100
		if lx == tot - 1:
			print('\rCompleted : {:.2f}%, on slot {}'.format(prog, fl), end=' ')
		else:
			print('\rCompleted : {:.2f}%, on slot {}\t{}'.format(prog, fl, fls[lx+1]), end=' ')
	fil.close()
	print()
	print('\nCreated File: {}'.format(dat+'/MapMatchResult.csv'))


if __name__ == '__main__':
	dat = 'probe_data_map_matching'
	p_id, d_t = loadTime(dat)
	p_head = loadProbeHeading(dat)
	p_speed = loadProbeSpeed(dat)
	p_alt = loadProbeAlt(dat)
	slots = timeSlots(p_id, d_t)
	p_x, p_y = loadProbeLatLong(dat)
	l_id, l_x, l_y = loadLinkLatLong(dat)
	dot = loadLinkDOT(dat)
	l_id, P1, P2, dots = createP1P2(l_id, l_x, l_y, dot)
	graph = loadLink(dat)[0]
	lidref = loadLinkIdentifiers(dat)
	theta = calculateTheta(P1, P2)
	lslopes = loadLinkSlope(dat)

	fls = ['1244899791.0']
	createMapMatch(dat, fls, slots, p_id, p_x, p_y, d_t, p_speed, p_head, p_alt, l_id, P1, P2, dots, theta, lslopes)
				# dots = []
				# uni, ind, count = np.unique(lids, return_index = True, return_counts=True)
				# for ix in range(lids.shape[0]):
				# 	if dot[lids[ix]] != 'B':
				# 		dots.append(dot[lids[ix]])
				# 	else:


				# print(P1)
					# for ix in range(ind.shape[0]):
						# p1.append(P1[ind[ix]])
			# print(coors, links)
			# r,c = createRoute(links, graph, lidref, dot, coors)
			# for pairs in c:
			# 	for pair in pairs:
			# 		plt.figure(1)
			# 		x1,y1 = map(float, pair[0].split(','))
			# 		x2,y2 = map(float, pair[1].split(','))
			# 		plt.plot(x1,y1, marker='+',markerfacecolor='b')
			# 		plt.plot(x2,y2, marker='+',markerfacecolor='b')
			# with open('file.csv','w') as f:
			# 	for pairs in r:
			# 		for ID in pairs:
			# 			f.write('{}\n'.format(ID))
			# pass
			# plt.show()
