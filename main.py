import imp
import shape as sp
imp.reload(sp)

import math
import itertools
import numpy as np

def simulate_volume(iterations, shape):
    points = np.random.uniform(-1,1,size=(3,iterations))
    samples = shape.clip(points)
    n_samples = sum(samples)
    return 8.0 * n_samples / iterations


def simulate_inv_projection(iterations, planes_normal, shape):
    points = np.random.uniform(-1,1,size=(3,iterations))
    
    n_samples = np.count_nonzero(shape.clip(points))
    v_original = 8.0 * n_samples / iterations

    i = 0
    volume = [None] * len(planes_normal)
    for pn in planes_normal:

        points = points.T[shape.clip_cylinder(points, pn)].T
        n_sample = points[0].size
        volume[i] = 8.0 * n_sample / iterations
        
        i+=1

    v = volume[-1]
    print(', '.join(('original volume:{0:f}','recovered:{1:f}','ratio:{2:f}')).format(v_original,v,v_original / v))

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
        sp.Sphere((-0.4,0,0),0.5) | sp.Sphere((0.4,0,0),0.5) | sp.Sphere([0,0.4,0],0.5)
}

print("simulating...")

phi = np.random.uniform(0,2 * np.pi, size=n_planes)
costheta = np.random.uniform(-1,1, size=n_planes)
theta = np.arccos(costheta)
planes_normal = np.vstack((np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta))).T

i = 0
for shape_name, shape in shapes.items():
    print('shape: ' + shape_name)
    points,_ = simulate_inv_projection(iterations, planes_normal, shape)

   
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    from IPython.display import Image

    points = points[:,:1000]
    x,y,z = points[0],points[1],points[2]
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
    marker = dict(color=np.where(shape.clip(points),0,1)+z*0.2,
            colorscale=[
                [0, 'rgb(226, 157, 38)'],
                [1, 'rgb(63, 127, 191)']],
            size=1.5,opacity=0.4)

    fig.append_trace(go.Scatter3d(x=x, y=y, z=z,
        mode='markers',
        marker=marker,
        scene='scene1'),1,1)

    '''fig.append_trace(go.Scatter3d(
        x=pnx, y=pny, z=pnz,
        mode='markers', marker=dict(size=5,opacity=0.9),
        scene='scene1'),
        1,3)'''

    proj_normal = np.array(((1,0,0),(0,1,0),(0,0,1)))
    for pn in proj_normal:
        projected = sp.project(points, pn, pn * (-1)).T
        fig.append_trace(go.Scatter3d(x=projected[0], y=projected[1], z=projected[2],
        mode='markers', marker=marker,
        scene='scene1'),1,1)


    fig['layout'].update(title=shape_name)
    fig['layout']['scene1'].update(scene)
    filename = 'test_' + shape_name.replace(' ','_')
    plotly.offline.plot(fig,filename='out\\' + '_' + filename + '.html',auto_open=False, show_link=False, validate=False)
    py.image.save_as(fig, filename='out\\' + filename + '.jpeg',width=1024,height=1024,)
    #Image('test.png')
    
    i+=1
