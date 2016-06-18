import imp
import shape as sp
imp.reload(sp)

import math
import itertools
import numpy as np
import os
import logging

logging.shutdown()
imp.reload(logging)
logger=logging.getLogger('main')
logger.addHandler(logging.FileHandler('./projection.log'))
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def uniform_sample(iterations):
    cuberoot=int(iterations**(1/3))
    steps=cuberoot*1j
    x=np.mgrid[-1:1.01:steps, -1:1.1:steps,-1.001:1:steps].reshape(3,-1).T
    rem=iterations-cuberoot**3
    if rem!=0:
        x=np.vstack((x, uniform_sample(rem)))
    return x

def monte_carlo_sample(iterations):
    return np.random.uniform(-1,1,size=(iterations, 3))

def simulate_inv_projection(iterations, planes_normal, shape):
    sample=uniform_sample
    points = sample(iterations)
    
    n_samples = np.count_nonzero(shape.clip(points))
    v_original = 8.0 * n_samples / iterations


    i = 0
    volume = [None] * len(planes_normal)
    for pn in planes_normal:

        points = points[shape.clip_cylinder(points, pn)]
        n_sample = points[0].size
        volume[i] = 8.0 * n_sample / iterations
        
        i+=1

    v = volume[-1]
    logger.info(', '.join(('original volume:{0:f}','recovered:{1:f}','ratio:{2:f}')).format(v_original,v,v_original / v))
    
    return points, v

iterations = 1000000
n_planes = 50


shapes = {
    'cube':sp.Cube(radius=0.5),
    'rect':sp.Rect([-0.5,-0.5,0],(1,0,0),(0,1,0)),
    'xy, yz, zx rect':
        sp.Rect([-0.5,-0.5,0],(1,0,0),(0,1,0)) | sp.Rect([0,-0.5,-0.5],(0,1,0),(0,0,1)) | sp.Rect([-0.5,0,-0.5],(1,0,0),(0,0,1)),
    'circle':sp.Circle((0,0,0),0.5,(1,0,0)),
    'xy, yz circle':
        sp.Circle((0,0,0),0.5,(1,0,0)) | sp.Circle((0,0,0),0.5,(0,1,0)),
    'xy, yz,zx circle':
        sp.Circle((0,0,0),0.5,(1,0,0)) | sp.Circle((0,0,0),0.5,(0,1,0)) | sp.Circle((0,0,0),0.5,(0,0,1)),
    'sphere':sp.Sphere(radius=0.5),
    '2 sphere intersection': 
        sp.Sphere((-0.25,0,0),0.5) & sp.Sphere((0.25,0,0),0.5),
    '2 sphere union':
        sp.Sphere((-0.4,0,0),0.5) | sp.Sphere((0.4,0,0),0.5),
    '3 sphere union':
        sp.Sphere((-0.4,0,0),0.5) | sp.Sphere((0.4,0,0),0.5) | sp.Sphere([0,0.8,0],0.5),
    'sliced cube faces': sp.PlaneSlice([0.5,0,0],[-1/math.sqrt(3)]*3)&
        (sp.Rect([0,0,0],(1,0,0),(0,1,0)) | 
        sp.Rect([0,0,0],(0,1,0),(0,0,1)) | 
        sp.Rect([0,0,0],(1,0,0),(0,0,1)))
}

#shapes={'xy, yz, zx rect':shapes['xy, yz, zx rect']}

logger.info("simulating...")

phi = np.random.uniform(0,2 * np.pi, size=n_planes)
costheta = np.random.uniform(-1,1, size=n_planes)
theta = np.arccos(costheta)
planes_normal = np.vstack((np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta))).T

i = 0
for shape_name, shape in shapes.items():
    logger.info('shape: ' + shape_name)
    points,_ = simulate_inv_projection(iterations, planes_normal, shape)

   
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    #from IPython.display import Image


    def takespread(sequence, num):
        length = float(len(sequence))
        for i in range(num):
            yield sequence[int(math.ceil(i * length / num))]
    if len(points)>5000:
        points = np.array(list(takespread(points, 5000)))
    x,y,z = points[:,0],points[:,1],points[:,2]
    pnx,pny,pnz = planes_normal[:,0],planes_normal[:,1],planes_normal[:,2]

    scene = dict(xaxis=dict(range=[-1,1]),
        yaxis=dict(range=[-1,1]),
        zaxis=dict(range=[-1,1]),
        camera=dict(up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1, y=1.5, z=0.5)))

    #fig=plotly.tools.make_subplots(2,3, specs=[[{'is_3d': True, 'rowspan':2,
    #'colspan':2},None, {'is_3d': True}],
     #                                          [None,None,None]])
    fig = plotly.tools.make_subplots(1,1, specs=[[{'is_3d': True}]])
    marker = dict(color=np.where(shape.clip(points),0,1)
                    +np.where(np.logical_xor.reduce((
                        np.mod(x,0.3)<=0.15,
                        np.mod(y,0.3)<=0.15,
                        np.mod(z,0.3)<=0.15)),0,1),
            colorscale=[
                [0, 'rgb(226, 157, 38)'],
                #[0.5, 'rgb(63, 127, 191)'],
                [1, 'rgb(63, 127, 191)']],
            size=2,opacity=1,
            line=dict(color='rgb(204,204,204)', width=0.2))
    
    fig.append_trace(go.Scatter3d(x=x, y=y, z=z,
        mode='markers',
        marker=marker.copy(),
        scene='scene1'),1,1)

    marker = dict(color=np.where(shape.clip(points),0,1),
            colorscale=[
                [0, 'rgb(226, 157, 38)'],
                #[0.5, 'rgb(63, 127, 191)'],
                [1, 'rgb(63, 127, 191)']],
            size=3,opacity=1)

    proj_normal = np.array(((1,0,0),(0,1,0),(0,0,1)))
    for pn in proj_normal:
        projected = sp.project(points, pn, pn * (-1)).T
        fig.append_trace(go.Scatter3d(x=projected[0], y=projected[1], z=projected[2],
        mode='markers', marker=marker,
        scene='scene1'),1,1)


    fig['layout'].update(title=shape_name)
    fig['layout']['scene1'].update(scene)
    filename = 'test_' + shape_name.replace(' ','_')
    directory='out\\'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plotly.offline.plot(fig,filename=directory + '_' + filename + '.html',auto_open=False)
    #py.image.save_as(fig, filename='out\\' + filename + '.jpeg',width=1024,height=1024,)
    #Image('test.png')
    i+=1
