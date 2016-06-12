import numpy as np

class Base:
    def __init__(self): pass
    def clip(s, x): pass
    def clip_cylinder(s, x, pn): pass

class Empty(Base):
    def __init__(self): pass
    def clip(s, x): 
        return np.zeros_like(x[0], dtype=bool)
    def clip_cylinder(s, x, pn): 
        return np.zeros_like(x[0], dtype=bool)
    
class Rect(Base):
    def __init__(s, origin, axisA, axisB):
        s.origin=origin
        s.axisA=axisA
        s.axisB=axisB
    def clip(s, x):
        Empty.clip(s, x)
    def clip_cylinder(s, x, pn):
        mat=np.vstack((s.axisA, s.axisB,-pn)).T
        if np.linalg.det(mat)==0: return Empty.clip_cylinder(s,x,pn)
        invmat=np.linalg.inv(mat)

        uvt=np.dot(invmat, (x.T-s.origin).T)
        u,v=uvt[0],uvt[1]
        return np.all((
            0<=u, u<=1,
            0<=v, v<=1),
            axis=0)

class Circle(Base):
    def __init__(s, origin, radius, normal):
        s.c=origin
        s.r=radius
        s.n=normal
    def clip(s, x):
        Empty.clip(s, x)
    def clip_cylinder(s, x, pn):
        x=x.T-s.c
        t=-np.dot(s.n, x.T)/np.dot(s.n, pn.T)
        return np.linalg.norm(x+np.outer(t,pn), axis=1)<=s.r

class Box(Base):
    def __init__(self, origin=np.array([-0.5,-0.5,-0.5]), size=np.array([1,1,1])):
        self.origin=origin
        self.size=size
        self.bound=np.vstack((origin, origin+size))
        xbasis=np.array([1,0,0])
        ybasis=np.array([0,1,0])
        zbasis=np.array([0,0,1])
        self.faces=[
            Rect(origin,size*ybasis,size*zbasis),
            Rect(origin+size*xbasis,size*ybasis,size*zbasis),
            Rect(origin,size*zbasis,size*xbasis),
            Rect(origin+size*ybasis,size*zbasis,size*xbasis),
            Rect(origin,size*xbasis,size*ybasis),
            Rect(origin+size*zbasis,size*xbasis,size*ybasis)
            ]
    def clip(s, x):
        return np.all((s.bound[0]<=x.T, x.T<=s.bound[1]), axis=(0,2))
    def clip_cylinder(s, x, pn):
        return np.any([f.clip_cylinder(x,pn) for f in s.faces], axis=0)

class Cube(Box):
    def __init__(self, center=np.array([0,0,0]), radius=0.5):
        Box.__init__(self, center-radius,radius*2)

class Sphere(Base):
    def __init__(self, origin=np.array([0,0,0]), radius=0.5):
        self.r=radius
        self.c=origin

    def clip(s, x):
        return np.linalg.norm(x.T-s.c, axis=1)<=s.r

    def clip_cylinder(s, x, pn):
        return np.linalg.norm(np.cross(x.T-(pn+s.c),x.T-s.c), axis=1)<=s.r

class Union(Base):
    def __init__(self, shapes):
        self.shapes=list(shapes)
    def clip(s, x):
        return np.any([shape.clip(x) for shape in s.shapes], axis=0)
    def clip_cylinder(s,x,pn):
        return np.any([shape.clip_cylinder(x,pn) for shape in s.shapes], axis=0)

class Intersection(Base):
    def __init__(self, shapes):
        self.shapes=list(shapes)
    def clip(s, x):
        return np.all([shape.clip(x) for shape in s.shapes], axis=0)
    def clip_cylinder(s,x,pn):
        return np.all([shape.clip_cylinder(x,pn) for shape in s.shapes], axis=0)