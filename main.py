import startup

def project(x,pn,pd):
	return x-pn*(np.dot(x,pn)-pd)

def clip_sphere(x,r=0.5):
	return np.linalg.norm(x, axis=0)<=r

def clip_cube(x,r=0.5):
    return np.all((np.abs(x[0])<=r, 
                  np.abs(x[1])<=r, 
                  np.abs(x[2])<=r),
                  axis=0)

def clip_column_sphere(x, pn,r=0.5):
	return np.linalg.norm(np.cross(x.T-pn,x.T), axis=1)<=r
		
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
    return np.any((
        clip_column_square_xy(x,pn,r,0.5),
        clip_column_square_xy(x,pn,r,-0.5),
        clip_column_square_yz(x,pn,r,0.5),
        clip_column_square_yz(x,pn,r,-0.5),
        clip_column_square_zx(x,pn,r,0.5),
        clip_column_square_zx(x,pn,r,-0.5)),
        axis=0)

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

def clip(x):
    return clip_cube(x, 0.5)
def clip_column(x,pn):
    return clip_column_cube(x, pn, 0.5)

for pn in planes_normal:
    
	points=points.T[clip_column(points, pn)].T
	n_sample=len(points)

	v=0
	v_=8.0*n_sample/n
	match[i-1]=1+(v-v_)/2**3
	
	i+=1


# plotting

points=points[:500]
x,y,z=points[0],points[1],points[2]

trace=go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        color=np.where(clip(points),0,1),
        colorscale=[
            [0, 'rgb(226, 157, 38)'],
            [1, 'rgb(63, 127, 191)']
            ],
        size=1,opacity=0.8)
    )
data=[trace]
layout=go.Layout(
    scene=dict(
        xaxis=dict(range=[-0.6,0.6]),
        yaxis=dict(range=[-0.6,0.6]),
        zaxis=dict(range=[-0.6,0.6]))
    )
fig=dict(data=data,layout=layout)
#py.plot(fig,filename='scatter-test')
print("plotting...")
plotly.offline.plot(fig,filename='scatter-test.html')