from load_data import *
import os
import json
import pickle
from collections import defaultdict

def generatePairs():
	pass

if __name__ == '__main__':
	dat = 'probe_data_map_matching'
	# p_id, d_t = loadTime(dat)
	# slots = timeSlots(p_id, d_t)
	p_x, p_y = loadProbeLatLong(dat)
	l_id, l_x, l_y = loadLinkLatLong(dat)
	dot = loadLinkDOT(dat)
	l_id, P1, P2 = createP1P2(l_id, l_x, l_y, dot)

	fls = ['1245061148.0.json']
	for fl in fls:
		slot = json.load(open('slot_cand/{}'.format(fl) ,'r'))
		for i,car in enumerate(slot):
			if i == 1:
				break
			coors = list(slot[car].keys())
			# print(coors)
			c2 = coors[:-1]
			c1 = coors[1:]
			for p1,p2 in zip(c1, c2):
				# print(slot[car][p1])
				pvid1 = [i[0] for i in slot[car][p1]]
				pvid2 = [i[0] for i in slot[car][p2]]
				print(pvid1, pvid2)
				pass
			# split = lambda x: [float(x.split(',')[0]), float(x.split(',')[1])]
			# coors = np.array([[split(x)] for x in coor])
			
			pass
		pass