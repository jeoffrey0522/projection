import imp
import shape as sp
imp.reload(sp)

import math
import itertools
import numpy as np


def project(x,pn,pd):
	return x-pn*(np.dot(x,pn)-pd)


print("simulating...")
n_planes=1000
iterations=1000000

planes_normal=[]
for _ in itertools.repeat(None, n_planes):
	pn=np.random.uniform(-1,1, size=3)
	pn=pn/np.linalg.norm(pn)
	planes_normal.append(pn)
#print(*planes_normal, sep='\n')


points=np.random.uniform(-1,1,size=(3,iterations))
n=points[0].size

#shape=sp.Cube(np.array([0.2,0.2,0.2]),0.4)
circle_normal=np.array([1,0.5,0.2])
circle_normal/=np.linalg.norm(circle_normal)

shape=sp.Union((
    sp.Circle(np.array([0,0,0]),0.5,np.array([1,0,0])),
    sp.Circle(np.array([0,0,0]),0.5,np.array([0,1,0])),
    sp.Circle(np.array([0,0,0]),0.5,np.array([0,0,1]))))

match=[0]*len(planes_normal)

i=1
for pn in planes_normal:
    
	points=points.T[shape.clip_cylinder(points, pn)].T
	n_sample=points[0].size

	v=0
	v_=8.0*n_sample/n
	match[i-1]=1+(v-v_)/2**3
	print('match rate: {0:%}, -log(1-x):{1:f}'.format(match[i-1],-math.log(1-match[i-1])))

	i+=1


# plotting
print("plotting...")
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

points=points[:,:6000]
x,y,z=points[0],points[1],points[2]

trace=go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        color=np.where(shape.clip(points),0,1),
        colorscale=[
            [0, 'rgb(226, 157, 38)'],
            [1, 'rgb(63, 127, 191)']
            ],
        size=1.5,opacity=0.4)
    )
data=[trace]
layout=go.Layout(
    scene=dict(
        xaxis=dict(range=[-1,1]),
        yaxis=dict(range=[-1,1]),
        zaxis=dict(range=[-1,1]))
    )
fig=dict(data=data,layout=layout)
#py.plot(fig,filename='scatter-test')
plotly.offline.plot(fig,filename='scatter_test.html')