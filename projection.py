

def project(x,pn,pd):
	return x-pn*(np.dot(x,pn)-pd)

def clip_sphere(x,r=0.5):
	return np.linalg.norm(x)<=r

def clip_cube(x,r=0.5):
    return np.all(np.abs(x[0])<=r, \
                  np.abs(x[1])<=r, \
                  np.abs(x[2])<=r)

def clip_column_sphere(x, pn,r=0.5):
	return np.linalg.norm(np.cross(x-(pn+c),x-c))<=r
		
def clip_column_square_xy(x, pn, r=0.5, h=0):
    t=(h-x[2])/pn[2]
    return np.logical_and(np.abs(x[0]+pn[0]*t)<=r, \
                          np.abs(x[1]+pn[1]*t)<=r)

def clip_column_square_yz(x, pn, r=0.5, h=0):
    t=(h-x[0])/pn[0]
    return np.logical_and(np.abs(x[1]+pn[1]*t)<=r, \
                          np.abs(x[2]+pn[2]*t)<=r)

def clip_column_square_zx (x, pn, r=0.5, h=0):
    t=(h-x[1])/pn[1]
    return np.logical_and(np.abs(x[0]+pn[0]*t)<=r, \
                          np.abs(x[2]+pn[2]*t)<=r)

def clip_column_cube(x, pn, r=0.5):
    return np.logical_or.reduce(
        (clip_column_square_xy(x,pn,r,0.5),
        clip_column_square_xy(x,pn,r,-0.5),
        clip_column_square_yz(x,pn,r,0.5),
        clip_column_square_yz(x,pn,r,-0.5),
        clip_column_square_zx(x,pn,r,0.5),
        clip_column_square_zx(x,pn,r,-0.5)))

print("simulating...")
planes_normal=[]
n_planes=4
for _ in itertools.repeat(None, n_planes):
	pn=np.random.uniform(-1,1, size=3)
	pn=pn/np.linalg.norm(pn)
	planes_normal.append(pn)
#print(*planes_normal, sep='\n')


iterations=50000
points=np.random.uniform(-1,1,size=iterations*3).reshape((3,iterations))
n=len(points)

match=[0]*n_planes
i=1

def clip_column(x,pn):
    return clip_column_cube(x, pn, 0.5)

for pn in planes_normal:
    
	points=points.T[clip_column(points, pn)].T
	n_sample=len(points)

	v=0
	v_=8.0*n_sample/n
	match[i-1]=1+(v-v_)/2**3
	
	i+=1


points=points[:1000]

trace=go.Scatter3d(x=points[0], 
                   y=points[1], 
                   z=points[2],
                   mode='markers',
                   marker=dict(
                       size=1.5,
                       opacity=0.8
                     ))
data=[trace]
layout=go.Layout(scene=dict(
    xaxis=dict(range=[-0.6,0.6]),
    yaxis=dict(range=[-0.6,0.6]),
    zaxis=dict(range=[-0.6,0.6]))
    )
fig=dict(data=data,layout=layout)
py.iplot(fig,filename='scatter-test')


"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax=plt.subplot2grid((2,2),(1,0),colspan=2)
ax.axis([1,n_planes,0,1.05])
ax.plot([1,1],[n_planes,1], label='expected')


ax.plot(range(1,n_planes+1),match)

ax = plt.subplot2grid((2,2),(0,0),projection='3d')
ax.axis([-0.5,0.5]*2)
ax.set_zlim(-0.5,0.5)
x,y,z=zip(*points[:100].T)
ax.scatter(x,y,z,s=3,c='b')

ax=plt.subplot2grid((2,2),(0,1),projection='3d')
x,y,z=zip(*planes_normal[:20])
ax.quiver(0,0,0,x,y,z,pivot='tail')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
		  
plt.show()"""