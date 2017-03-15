from load_data import *
import numpy as np
import os
import sys
import json
from collections import defaultdict, OrderedDict
from matplotlib import pyplot as plt

def findPath(a,b):
	pass

def generatePairs(P1, P2, d, ref):
	for i in range(P1.shape[0]):
		p1 = d[P1[i]]
		p2 = d[P2[i]]

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
	# theta[ind_t] = (2.5*pi - arctan(delta[ind_t,1]/delta[ind_t,0])) * 180./pi
	# theta[ind_f] = (0.5*pi - arctan(delta[ind_f,1]/delta[ind_f,0])) * 180./pi
	
	# # ind = np.where(np.isnan(theta))[0]
	# # # print(ind)

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
	# return np.array([])

# def linkPairs()

if __name__ == '__main__':
	dat = 'probe_data_map_matching'
	l_id, dist = loadLinkLength(dat)

	idref = loadLinkIdentifiers(dat)

	lid, l_x, l_y = loadLinkLatLong(dat)

	phead = loadProbeHeading(dat)

	dot = loadLinkDOT(dat)
	lid, P1, P2 = createP1P2(lid, l_x, l_y, dot)

	print(P1.shape)
	# for i, x in enumerate(idref.items()):
	# 	if i == 10:
	# 		break
	# 	k,v = x
	# 	print('{}: {}'.format(k,v))

	# graph = loadLink(dat)[1]
	# for x in graph:
	# 	print(graph[x])
	# 	break
	# lID, ind, count = np.unique(l_id,return_index = True, return_counts=True)
	# for i in range(lID.shape[0]):
	# 	X = l_x[ind[i]: ind[i]+count[i]]
	# 	Y = l_y[ind[i]: ind[i]+count[i]]
	# 	# X = l_x[np.where(lid == l_id[i])]
	# 	# Y = l_y[np.where(lid == l_id[i])]
	# 	plt.plot(X,Y,c='green',linestyle='-',linewidth=2,marker='o',mfc='red',markersize=2)
	pvid = set()
	# fli = ['1199426401.0.json','1244811116.0.json','1244812317.0.json']
	fli = ['1245061148.0.json']
	for fl in fli:
		slot = OrderedDict(json.load(open('slot_cand/{}'.format(fl),'r')))
	# P = np.vstack((l_x,l_y))
	# P = P.T

	# # f = open('Link.csv','w')
	# # for i in range(P.shape[0]):
	# # 	f.write('{},{},{}\n'.format(lid[i],P[i,0],P[i,1]))
	# # f.close()
	# # np.savetxt(open(dat+'/Link.csv','wb'),P,fmt='%.10f', delimiter=',')

	# P1 = P[:-1]
	# l1 = lid[:-1]
	# P2 = P[1:]
	# l2 = lid[1:]

	# ind = np.where(l1 == l2)
	# print(ind[0].shape)
	# lid = l1[ind]
	# P1 = P1[ind]
	# P2 = P2[ind]
	
		for i,car in enumerate(slot):
		# if i == 1:
		# 	break
		# print(slot[car])
		# P1 = np.array(list(slot[car].keys()))
		# P2 = P1.copy()
		# print(P1.shape)
		# P2 = P1[1:]
		# P1 = P1[:-1]
		# print(P1, P2)
		# generatePairs(P1,P2,slot[car], idref)
		# pvid = set()
			for j,coor in enumerate(slot[car]):
			# print(len(slot[car][coor]))
			# if j == 10:
			# 	break
				x,y = coor.split(',')
				x = float(x)
				y = float(y)

				rxMin = x - 0.0001
				rxMax = x + 0.0001
				ryMin = y - 0.0001
				ryMax = y + 0.0001

				ind = np.where((P1[:,0] >= rxMin) & (P1[:,0] <= rxMax) & (P1[:,1] >= ryMin) & (P1[:,1] <= ryMax))[0]
				P1_f = P1[ind,:]
				P2_f = P2[ind,:]
				lid_f = lid[ind]

				temp, ps = calculatePD(P1_f,P2_f,(x,y))
				ind = temp.argsort()
				temp = temp[ind]
				l = lid_f[ind]
				p1 = P1_f[ind,:]
				p2 = P2_f[ind,:]
				ps = ps[ind]
			# if temp.shape[0] > 0:
			# 	pvid.add(l[0])
			# 	print('({},{}); {}; PVID:{}, Ppoint:({},{}), {},{}'.format(x,y,temp[0],l[0],ps[0,0],ps[0,1],p1[0,:],p2[0,:]))
			# print('{},{}: PVID: {}, Distance: {:.4f}, {},{}'.format(x,y, l[0], temp[0], P1[0], P2[0]))
			# print(x, y)
				plt.figure(1)
				plt.plot(x,y,marker='^', color='blue',markersize=5)
				for m,k in enumerate(slot[car][coor]):
					print(k[0])
					pvid.add(k[0])
			# 	if m == 1:
			# 		break
			# if l.shape[0] > 0:
			# 	id = l[0]
			# 	X = l_x[np.where(lid == id)[0]]
			# 	Y = l_y[np.where(lid == id)[0]]
			# 	plt.figure(2)
			# 	plt.plot(X,Y, color='green',marker='o',mfc='red',linestyle='-',linewidth=2)

	f = open('file.csv','w')
	for x in pvid:
		f.write('{}\n'.format(x))
	f.close()
	# # # plt.axes([51.1718, 10.9567, 0.02, 0.043])
	plt.xlabel('Latitude')
	plt.ylabel('Longitude')
	# plt.figure(2)
	# plt.xlabel('Latitude')
	# plt.ylabel('Longitude')
	plt.show()
	# # 		print()
	# # 	print()
	# # # 	pass


	# # path = findPath(graph,[])
	# # l_id, l_x, l_y = loadLinkLatLong(dat)
	# # lid = lid[:-1]

	


	# # P1 = np.array([[4,8],
	# # 	[10,1],
	# # 	[3,9]], dtype=np.float64)

	# # P2 = np.array([[10,1],
	# # 	[3,9],
	# # 	[5,2]],dtype=np.float64)
	# ER = 6371.
	# x3, y3 = 51.1724748183,10.9884865489
	# p3 = np.array([x3,y3])
	# dLat = (0.05/ER)*(180/np.pi)
	# dLong = (0.05/ER)*(180/np.pi)/np.cos(x3*np.pi/180)
	# # print(dLat, dLong)
	# rxMin = x3 - dLat
	# rxMax = x3 + dLat
	# ryMin = y3 - dLong
	# ryMax = y3 + dLong
	# print(rxMin, rxMax, ryMin, ryMax)
	# print()
	# # ind = np.where(((P1[:,0] >= (p3[0]-dLat)) & (P1[:,0] <= (p3[0]+dLat))) & ((P1[:,1] >= (p3[1]-dLong)) & (P1[:,1] <= (p3[1]+dLong))))
	# ind = np.where((P1[:,0] >= rxMin) & (P1[:,0] <= rxMax) & (P1[:,1] >= ryMin) & (P1[:,1] <= ryMax))[0]
	# print(ind.shape)
	# p1_f = P1[ind,:]
	# p2_f = P2[ind,:]
	# lid_f = lid[ind]
	# # print(p1_)

	# # ind = np.where((P1[:,1] >= ryMin) & (P1[:,1] <= ryMax))
	# # print(ind[0].shape)

	# x,p = calculatePD(p1_f,p2_f,(x3,y3))
	# # print(x, p)
	# ind = x.argsort()
	# x = x[ind]
	# l = lid_f[ind]
	# p1 = p1_f[ind]
	# p2 = p2_f[ind]
	# p = p[ind]
	# print('{}; PVID:{}, Ppoint:({},{}), {},{}'.format(x[0],l[0],p[0,0],p[0,1],P1[0,:],P2[0,:]))
	# for i in range(10):
	# 	print('{}; PVID:{}, Ppoint:({},{}), {},{}'.format(x[i],l[i],p[i,0],p[i,1],P1[i,:],P2[i,:]))
		# print('Dist: {}, PVID: {}, p1: {}, p2: {}'.format(x[i], l[i], P1[i], P2[i]))

	# p1 = np.array([[51.1728399, 10.98829]])
	# p2 = np.array([[51.1730854, 10.9881512]])
	# t1 = calculateTheta(P1,P2)
	# # # print(t)
	# ind = np.where(t1 > 360)[0]
	# print(ind.shape)
	# # p2 = np.array([[51.1728399, 10.98829]])
	# # p1 = np.array([[51.1730854, 10.9881512]])
	# t2 = calculateTheta(P2,P1)
	# # print(t)
	# ind = np.where(t2 < 0)[0]
	# print(ind.shape)
	# ind = np.where(t1 == t2)[0]
	# print(ind.shape)
	# print(dot[str(762466209)])