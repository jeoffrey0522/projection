import numpy as np

def project(x,pn,c):
        x=x.T-c
        t=-np.dot(pn, x.T)
        return x+np.outer(t,pn)+c

class Base:
    def __init__(self): raise NotImplementedError()
    def clip(self, x): raise NotImplementedError()
    def clip_cylinder(self, x, pn): raise NotImplementedError()

    def __or__(self, other): return Union((self, other))
    def __and__(self, other): return Intersection((self, other))
    def __sub__(self,other): return Difference(self, other)

class Empty(Base):
    def __init__(self): pass
    def clip(self, x): return np.zeros_like(x[0], dtype=bool)
    def clip_cylinder(self, x, pn): return np.zeros_like(x[0], dtype=bool)
    
class Rect(Base):
    def __init__(self, origin, axisA, axisB):
        self.origin=np.array(origin)
        self.axisA=np.array(axisA)
        self.axisB=np.array(axisB)
    def clip(self, x): return Empty.clip(self, x)
    def clip_cylinder(self, x, pn):
        mat=np.vstack((self.axisA, self.axisB,-pn)).T
        if np.linalg.det(mat)==0: return Empty.clip_cylinder(self,x,pn)
        invmat=np.linalg.inv(mat)

        uvt=np.dot(invmat, (x.T-self.origin).T)
        u,v=uvt[0],uvt[1]
        return np.all((
            0<=u, u<=1,
            0<=v, v<=1),
            axis=0)

class Circle(Base):
    def __init__(self, origin, radius, normal):
        self.c=np.array(origin)
        self.r=radius
        self.n=np.array(normal)
    def clip(self, x):
        return Empty.clip(self, x)
    def clip_cylinder(self, x, pn):
        x=x.T-self.c
        div=np.dot(self.n, pn.T)
        div=div if div!=0 else 0.001
        t=-np.dot(self.n, x.T)/div
        return np.logical_and(
            np.isfinite(t),
            np.linalg.norm(x+np.outer(t,pn), axis=1)<=self.r
            )

class Box(Base):
    def __init__(self, origin=[-0.5,-0.5,-0.5], size=[1,1,1]):
        self.origin=np.array(origin)
        self.size=np.array(size)
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
    def clip(self, x):
        return np.all((self.bound[0]<=x.T, x.T<=self.bound[1]), axis=(0,2))
    def clip_cylinder(self, x, pn):
        return np.any([f.clip_cylinder(x,pn) for f in self.faces], axis=0)

class Cube(Box):
    def __init__(self, center=[0,0,0], radius=0.5):
        Box.__init__(self, np.array(center)-radius,radius*2)

class Sphere(Base):
    def __init__(self, origin=[0,0,0], radius=0.5):
        self.c=np.array(origin)
        self.r=radius

    def clip(self, x):
        return np.linalg.norm(x.T-self.c, axis=1)<=self.r

    def clip_cylinder(self, x, pn):
        return np.linalg.norm(np.cross(x.T-(pn+self.c),x.T-self.c), axis=1)<=self.r

class Union(Base):
    def __init__(self, shapes):
        self.shapes=list(shapes)
    def clip(self, x):
        return np.any([shape.clip(x) for shape in self.shapes], axis=0)
    def clip_cylinder(self,x,pn):
        return np.any([shape.clip_cylinder(x,pn) for shape in self.shapes], axis=0)

class Intersection(Base):
    def __init__(self, shapes):
        self.shapes=list(shapes)
    def clip(self, x):
        return np.all([shape.clip(x) for shape in self.shapes], axis=0)
    def clip_cylinder(self,x,pn):
        return np.all([shape.clip_cylinder(x,pn) for shape in self.shapes], axis=0)

class Difference(Base):
    def __init__(self, shape1, shape2):
        self.shape=Intersection((shape1,Inverse(shape2)))

class Inverse(Base):
    def __init__(self, shape):
        self.shape=shape
    def clip(self, x):
        return np.logical_not(self.shape.clip(x), dtype=bool)
    def clip_cylinder(self, x):
        return np.logical_not(self.shape.clip_cylinder(x, pn), dtype=bool)