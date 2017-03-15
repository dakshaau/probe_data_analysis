from load_data import *
import os
import json
import pickle
from collections import defaultdict
from datetime import datetime

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

if __name__ == '__main__':
	dat = 'probe_data_map_matching'
	p_id, d_t = loadTime(dat)
	slots = timeSlots(p_id, d_t)
	p_x, p_y = loadProbeLatLong(dat)
	l_id, l_x, l_y = loadLinkLatLong(dat)
	dot = loadLinkDOT(dat)
	l_id, P1, P2 = createP1P2(l_id, l_x, l_y, dot)
	graph = loadLink(dat)[0]
	lidref = loadLinkIdentifiers(dat)
	# print(p_id.shape[0])
	# print(d_t.shape[0])
	# print(p_x.shape[0])
	# print(p_y.shape[0])
	# g = {1: [2,4], 4: [3,1], 3: [5,6], 2: [], 5: [], 6: []}
	# visited = defaultdict(lambda: False)
	# result = defaultdict(lambda: False)
	# x = isChild(1,5,g,visited,result)
	# print(x)

	fls = ['1245061148.0.json']
	for fl in fls:
		slot = json.load(open('slot_cand/{}'.format(fl) ,'r'))
		slot_ID = fl.split('.json')[0]
		for i,car in enumerate(slot):
			# if i == 1:
			# 	break
			times = sorted(slots[slot_ID][car])
			ind = np.where((p_id == car) & ((d_t >= times[0]) & (d_t <= times[-1])))[0]
			x = p_x[ind]
			y = p_y[ind]
			# print(car)
			# print(datetime.fromtimestamp(slots[slot_ID][car][0]))
			# print(datetime.fromtimestamp(slots[slot_ID][car][-1]))
			coors = ['{},{}'.format(str(x[k]),str(y[k])) for k in range(x.shape[0])]
			c = []
			[c.append(k) for k in coors if k not in c]
			del coors
			coors = c
			# print(len(coors1))
			# coors2 = list(slot[car].keys())
			# print(len(coors2))
			# print(sorted(coors1) == sorted(coors2))
			# print(sorted(coors1)[:5])
			# print(sorted(coors2)[:5])
			# print()
			# continue
			c2 = coors[:-1]
			c1 = coors[1:]
			del coors
			for p1,p2 in zip(c1, c2):
				# print(slot[car][p1])
				pvid1 = [i[0] for i in slot[car][p1]]
				pvid2 = [i[0] for i in slot[car][p2]]
				pairs = generatePairs(pvid1, pvid2)
				# print(pvid1, pvid2)
				for p1, p2 in pairs:
					print(p1, p2)
					print(isConnected(p1, p2, graph, lidref, dot))
				pass
			# split = lambda x: [float(x.split(',')[0]), float(x.split(',')[1])]
			# coors = np.array([[split(x)] for x in coor])
			
			pass
		pass