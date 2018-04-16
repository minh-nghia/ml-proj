from geopy.geocoders import Nominatim
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax._axis3don = False


def get_coor(location):
    loc = geolocator.geocode(location)
    lat = loc.latitude
    long = loc.longitude
    
    return lat, long


def coor_to_xyz(lat, long):
    x = np.cos(np.pi*lat/180) * np.cos(np.pi*long/180)
    y = np.cos(np.pi*lat/180) * np.sin(np.pi*long/180)
    z = np.sin(np.pi*lat/180)

    return x, y, z

geolocator = Nominatim()

global_ = 0, 0, 0
ax.scatter(*global_)

lat1, long1 = get_coor('Montreal')
northamerica_northeast1 = coor_to_xyz(lat1, long1)
print('"northamerica-northeast1": ', northamerica_northeast1)
ax.scatter(*northamerica_northeast1)

lat2, long2 = get_coor('Council Bluffs')
us_central1 = coor_to_xyz(lat2, long2)
print('"us-central1": ', us_central1)
ax.scatter(*us_central1)

lat3, long3 = get_coor('The Dalles')
us_west1 = coor_to_xyz(lat3, long3)
print('"us-west1": ', us_west1)
ax.scatter(*us_west1)

lat4, long4 = get_coor('Ashburn, Virginia')
us_east4 = coor_to_xyz(lat4, long4)
print('"us-east4": ', us_east4)
ax.scatter(*us_east4)

lat5, long5 = get_coor('Moncks Corner')
us_east1 = coor_to_xyz(lat5, long5)
print('"us-east1": ', us_east1)
ax.scatter(*us_east1)

us = coor_to_xyz(np.mean([lat2, lat3, lat4, lat5]), np.mean([long2, long3, long4, long5]))
print('"us": ', us)
ax.scatter(*us)

lat6, long6 = get_coor('Sao Paulo')
southamerica_east1 = coor_to_xyz(lat6, long6 )
print('"southamerica-east1": ', southamerica_east1)
ax.scatter(*southamerica_east1)

lat7, long7 = get_coor('St. Ghislain')
europe_west1 = coor_to_xyz(lat7, long7)
print('"europe-west1": ', europe_west1)
ax.scatter(*europe_west1)

lat8, long8 = get_coor('London')
europe_west2 = coor_to_xyz(lat8, long8)
print('"europe-west2": ', europe_west2)
ax.scatter(*europe_west2)

lat9, long9 = get_coor('Frankfurt')
europe_west3 = coor_to_xyz(lat9, long9)
print('"europe-west3": ', europe_west3)
ax.scatter(*europe_west3)

lat10, long10 = get_coor('Eemshaven')
europe_west4 = coor_to_xyz(lat10, long10)
print('"europe-west4": ', europe_west4)
ax.scatter(*europe_west4)

eu = coor_to_xyz(np.mean([lat7, lat8, lat9, lat10]), np.mean([long7, long8, long9, long10]))
print('"eu": ', eu)
ax.scatter(*eu)

lat11, long11 = get_coor('Mumbai')
asia_south1 = coor_to_xyz(lat11, long11)
print('"asia-south1": ', asia_south1)
ax.scatter(*asia_south1)

lat12, long12 = get_coor('Jurong West')
asia_southeast1 = coor_to_xyz(lat12, long12)
print('"asia-southeast1": ', asia_southeast1)
ax.scatter(*asia_southeast1)

lat13, long13 = get_coor('Changhua County')
asia_east1 = coor_to_xyz(lat13, long13)
print('"asia-east1": ', asia_east1)
ax.scatter(*asia_east1)

lat14, long14 = get_coor('Tokyo')
asia_northeast1 = coor_to_xyz(lat14, long14)
print('"asia-northeast1": ', asia_northeast1)
ax.scatter(*asia_northeast1)

lat15, long15 = get_coor('Sydney')
australia_southeast1 = coor_to_xyz(lat15, long15)
print('"australia-southeast1": ', australia_southeast1)
ax.scatter(*australia_southeast1)

lat16, long16 = get_coor('USA')
us_country = coor_to_xyz(lat16, long16)
print('"us": ', us_country)
ax.scatter(*us_country)

lat17, long17 = get_coor('Ireland')
ie_country = coor_to_xyz(lat17, long17)
print('"ie": ', ie_country)
ax.scatter(*ie_country)

lat18, long18 = get_coor('Great Britain')
gb_country = coor_to_xyz(lat18, long18)
print('"gb": ', gb_country)
ax.scatter(*gb_country)

lat19, long19 = get_coor('India')
in_country = coor_to_xyz(lat19, long19)
print('"in": ', in_country)
ax.scatter(*in_country)

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.sin(u)*np.cos(v)
y = np.sin(u)*np.sin(v)
z = np.cos(u)
ax.plot_wireframe(x, y, z, color='gray')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
