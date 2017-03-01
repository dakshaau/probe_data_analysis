from load_data import *
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict

def createCandidate():
	pass

if __name__ == '__main__':
	dat = 'probe_data_map_matching'
	p_id, d_t = loadTime(dat)
	slots = timeSlots(p_id, d_t)
	p_x, p_y = loadProbeLatLong(dat)
	l_x, l_y = loadLinkLatLong(dat)
	p_speed = loadProbeSpeed(dat)
	p_head = loadProbeHeading(dat)
	print(p_speed[:10])
	print(p_head[:10])
