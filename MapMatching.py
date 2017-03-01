from load_data import loadProbeLatLong, loadLinkLatLong, timeSlots, loadTime
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
	l_x, l_y = loadLinkLatLong(dat)
