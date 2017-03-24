Probe Data Analysis for Road Slope

Language: Python 3.5

To run the main code:
 - Make sure probe and link data files are in a folder named 'probe_data_map_matching'
 - Execute MapMatching.py to get the results


 MatchedPoints Record Format:

	sampleID, dateTime, latitude, longitude, altitude, speed, heading, linkPVID, direction, distFromRef, distFromLink

		sampleID	is a unique identifier for the set of probe points that were collected from a particular phone.
		dateTime	is the date and time that the probe point was collected.
		latitude	is the latitude in decimal degrees.
		longitude	is the longitude in decimal degrees.
		altitude	is the altitude in meters.
		speed		is the speed in MPH.
		heading		is the heading in degrees.
		linkPVID	is the published versioned identifier for the link.
		direction	is the direction the vehicle was travelling on thelink (F = from ref node, T = towards ref node).
		distFromRef	is the distance from the reference node to the map-matched probe point location on the link in decimal meters.
		distFromLink	is the perpendicular distance from the map-matched probe point location on the link to the probe point in decimal meters.