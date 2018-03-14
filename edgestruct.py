# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:44:55 2017

@author: julienolivier

Python module for managing meshes through edge structure.
Version 1

Version 1 : manages only 2D meshes.
Provides:
class Mesh : internal managing of meshes
function rectangle:  creates cartesian regular quadrangle meshes
function readmesh: creates mesh from .msh structure
function plotdiscrete: plots a discrete function on a mesh
"""

import warnings as wng
import time

import matplotlib.pylab as plt
import matplotlib.collections as coll
import numpy as np
import numpy.linalg as la
import scipy.sparse as spsp
import scipy.sparse.linalg as spla

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.close("all")
pi = np.pi
cos = np.cos
sin = np.sin



class MeshError(Exception):
    pass


class FormatError(MeshError):
    def __init__(self, s, level):
        self.file = s
        self.lvl = level

    def __str__(self):
        what_failed = {0: 'Before first line',
                       1: 'First line',
                       2: 'Vertex description',
                       3: 'Vertex number of edge description',
                       4: 'Volume description'}
        output = 'An error occurred while loading file: '+self.file+' '
        output += 'in the following part of the file\n'
        output += what_failed[self.lvl]
        return output


class UnstructuredError(MeshError):
    def __init__(self, s):
        self.name = s

    def __str__(self):
        output = 'Cannot treat non triangular meshes yet.'
        return output


def wedge(u, v):
    """wedge(u, v)
    Compute the wedge product of two 2D vectors"""
    if u.ndim > 1:
        return u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
    else:
        return u[0]*v[1]-u[1]*v[0]


def triarea(a, b, c):
    """triarea(a, b, c)
    Compute the area of a triangle given by three vertices"""
    return 0.5*abs(wedge(b-a, c-a))


def orthocenter(a, b, c):
    """orthocenter(a, b, c)
    Compute the orthocenter of a triangle given by three vertices"""
    o = a+0.5*(b-a)+(0.5*np.inner(c-b, c-a)/wedge(c-b, b-a)
                     * np.array([b[1]-a[1], -(b[0]-a[0])]))
    return o


def harmavg(a, b, alpha, beta):
    """harmavg(a, b, alpha, beta)
    Compute the harmonic average of a, b with weights alpha and beta
    """
    return (alpha+beta)*a*b/(alpha*a+beta*b)


class Mesh():
    """The class to handle a mesh via its edges description.

    Attributes :

    centres: positions of the centers of the volumes
    nvol: number of volumes
    vert: positions of the vertices
    nvert: number of vertices
    mx, Mx, my, My : window containing the mesh. Mesh can be drawn in
        [mx, Mx]x[my, My]
    inner: the inner edges
    nin: the number of inner edges
    bnd: the boundary edges
    nbnd: the number of boundary edges
    Kin, Lin: arrays such that the inner edge which has vertices at
        vert[inner[0, i]] and vert[inner[1, i]] bounds the volume whose centers
        are centres[Kin[i]] and centres[Lin[i]]
    Kbnd: array such that the boundary edge which has vertices at
        vert[bnd[0, j]] and vert[bnd[1, j]] bounds the volume whose center is
        centre[Kin[i]].
    xs_i: the intersection of [K, L] and [s1, s2]
    xs_b: the orthogonal projection of K on [s1, s2]
    dKL: the distance between K and L
    dKs_i: distance between K and xs for inner edges
    dKs_b: distance between K and xs for boundary edges
    dLs: distance beteen L and xs.
    mes_i: length of the inner edges
    mes_b: length of th boundary edges
    m1KS_i: area of the triangle with vertices s1, K, xs of inner edges
    m1KS_b: area of the triangle with vertices s1, K, xs of boundary edges
    m2KS_i: area of the triangle with vertices s2, K, xs of inner edges
    m2KS_b: area of the triangle with vertices s2, K, xs of boundary edges
    m1LS: area of the triangle with vertices s1, L, xs.
    m2LS: area of the triangle with vertices s2, L, xs.
    nKs_i: normalized vector from K to xs of inner edges
    nKs_b: normalized vector from K to xs of boundary edges

    -------------------------------------------------------
    Methods :

    create_geometry : creates the geometrical data
    compute_vol : computes the volumes of the elements in the mesh and return
        them
    plotmesh : plots a representation of the mesh
    """

    def __init__(self, c, v, ai, ae, Ki, Li, Kb):
        """__init__(self, c, v, ai, ae, Ki, Li, Kb)
        Initialize an edge structure.
        Mesh(c, v, ai, ae, Ki, Li, Kb)

        c: the array of centers of shape (nvol, 2).
        v: the array of vertices of shape (nvert, 2).
        ai: the array of inner edges of shape (2, smthg).
        ae: the array of boundary edges of shape (2, smthg).
        Ki, Li: the arrays of volumes bounded by inner edges.
        Kb: the array of volumes bounded by boundary edges.
        """
        self.centres = c
        self.nvol = c.shape[0]
        self.vert = v
        self.nvert = v.shape[0]
        self.Mx = max(np.amax(self.vert[:, 0]), np.amax(self.centres[:, 0]))
        self.mx = min(np.amin(self.vert[:, 0]), np.amin(self.centres[:, 0]))
        self.My = max(np.amax(self.vert[:, 1]), np.amax(self.centres[:, 1]))
        self.my = min(np.amin(self.vert[:, 1]), np.amin(self.centres[:, 1]))
        self.inner = ai
        self.nin = ai.shape[1]
        self.bnd = ae
        self.nbnd = ae.shape[1]
        self.Kin = Ki
        self.Lin = Li
        self.Kbnd = Kb
        self.create_geometry()

    def create_geometry(self):
        """ Create all the geometric information related to the mesh.

        This method will set geometric attributes for the edge.
        """
        s_i = self.vert[self.inner]
        s_b = self.vert[self.bnd]
        k_i = self.centres[self.Kin]
        l_i = self.centres[self.Lin]
        k_b = self.centres[self.Kbnd]
        xs_b = s_b[0]-(np.tile(np.einsum('ij,ij->i', s_b[1]-s_b[0], s_b[0]-k_b)
                       / la.norm(s_b[1]-s_b[0], axis=1)**2, (2, 1)).T
                       * (s_b[1]-s_b[0]))
        xs_i = k_i+(np.tile(wedge(s_i[1]-s_i[0], s_i[1]-k_i)
                    / wedge(s_i[1]-s_i[0], l_i-k_i), (2, 1)).T
                    * (l_i-k_i))
        self.dKL = la.norm(l_i-k_i, axis=1)
        self.dLs = la.norm(xs_i-l_i, axis=1)
        self.m1Ls = triarea(s_i[0], l_i, xs_i)
        self.m2Ls = triarea(s_i[1], l_i, xs_i)
        self.xs_i = xs_i
        self.xs_b = xs_b
        self.mes_i = la.norm(s_i[1]-s_i[0], axis=1)
        self.mes_b = la.norm(s_b[1]-s_b[0], axis=1)
        self.dKs_i = la.norm(xs_i-k_i, axis=1)
        self.dKs_b = la.norm(xs_b-k_b, axis=1)
        self.m1Ks_i = triarea(s_i[0], k_i, xs_i)
        self.m1Ks_b = triarea(s_b[0], k_b, xs_b)
        self.m2Ks_i = triarea(s_i[1], k_i, xs_i)
        self.m2Ks_b = triarea(s_b[1], k_b, xs_b)
        self.nKs_i = (xs_i-k_i)/np.tile(la.norm(xs_i-k_i, axis=1), (2, 1)).T
        self.nKs_b = (xs_b-k_b)/np.tile(la.norm(xs_b-k_b, axis=1), (2, 1)).T

    def plotmesh(self, **how):
        """ Draw a plot of the mesh.

        Inner edges in blue lines. Boundary edges in magenta lines. Centers as
        circles. Vertices as squares. x_sigma as x. nKs as arrows.
        """

        ax = plt.gca()
        ax.set_xlim((1.1*self.mx-0.1*self.Mx, -0.1*self.mx+1.1*self.Mx))
        ax.set_ylim((1.1*self.my-0.1*self.My, -0.1*self.my+1.1*self.My))
        ax.set_aspect('equal')
        s_i = self.vert[self.inner]
        s_b = self.vert[self.bnd]
        X_i = np.stack((s_i[0, :, 0], s_i[1, :, 0]))
        Y_i = np.stack((s_i[0, :, 1], s_i[1, :, 1]))
        X_b = np.stack((s_b[0, :, 0], s_b[1, :, 0]))
        Y_b = np.stack((s_b[0, :, 1], s_b[1, :, 1]))
        plt.plot(X_i, Y_i, '-b')
        plt.plot(X_b, Y_b, '-m')
        plt.plot(self.vert[:, 0], self.vert[:, 1], 's', **how)
        plt.plot(self.centres[:, 0], self.centres[:, 1], 'go', **how)
        plt.plot(self.xs_b[:, 0], self.xs_b[:, 1], 'rx', **how)
        plt.plot(self.xs_i[:, 0], self.xs_i[:, 1], 'rx', **how)
        plt.quiver(self.xs_i[:, 0], self.xs_i[:, 1],
                   self.nKs_i[:, 0], self.nKs_i[:, 1])
        plt.quiver(self.xs_b[:, 0], self.xs_b[:, 1],
                   self.nKs_b[:, 0], self.nKs_b[:, 1])

    def compute_vol(self):
        """compute_vol(self)
        Computes the volumes of all the elements in the mesh

        Returns:
            V : an (self.nvol) array such that V[i] is the volume of the
            element whose center is self.centres[i]"""
        V = np.zeros(self.nvol)
        for i in range(self.nvol):
            Iin = np.where(self.Kin == i)
            Jin = np.where(self.Lin == i)
            Ibnd = np.where(self.Kbnd == i)
            V[i] = (np.sum(self.m1Ks_i[Iin])
                    + np.sum(self.m2Ks_i[Iin])
                    + np.sum(self.m1Ls[Jin])
                    + np.sum(self.m2Ls[Jin])
                    + np.sum(self.m1Ks_b[Ibnd])
                    + np.sum(self.m2Ks_b[Ibnd]))
        return V


def rectangle(nx, ny, a=0, b=1, c=0, d=1):
    """rectangle(nx, ny, a=0, b=1, c=0, d=1)
    Create the edge structure of the quadrangle mesh of a rectangle.

    Rectangle to mesh is [a, b]x[c, d]. Number of volumes by column is nx, by
    line is ny. Total number of volumes is thus nx*ny. Number of vertices is
    (nx+1)*(ny+1).
    """
    conc = np.concatenate
    arng = np.arange
    hx = (b-a)/nx
    hy = (d-c)/ny
    Ki = np.zeros((nx-1)*(ny-1), dtype='int64')
    Li = np.zeros((nx-1)*(ny-1), dtype='int64')

    xv = a+hx*arng(nx+1)
    yv = c+hy*arng(ny+1)
    v = np.stack((np.tile(xv, ny+1), np.repeat(yv, nx+1)), axis=1)

    xc = a+hx*(0.5+np.arange(nx))
    yc = c+hy*(0.5+arng(ny))
    cent = np.stack((np.tile(xc,  ny), np.repeat(yc, nx)), axis=1)

    eb_bot = conc((arng(nx).reshape(1, nx),
                   arng(1, nx+1).reshape(1, nx)))
    Kb_bot = arng(nx)
    eb_top = conc((arng(ny*(nx+1), (nx+1)*ny+nx).reshape(1, nx),
                   arng(1+ny*(nx+1), (nx+1)*(ny+1)).reshape(1, nx)))
    Kb_top = (ny-1)*nx+arng(nx)
    eb_lft = conc((arng(0, ny*(nx+1), nx+1).reshape(1, ny),
                   arng(nx+1, (ny+1)*(nx+1), nx+1).reshape(1, ny)))
    Kb_lft = nx*arng(ny)
    eb_rgt = conc((arng(nx, nx+ny*(nx+1), nx+1).reshape(1, ny),
                   arng(2*nx+1, (ny+1)*(nx+1)+nx, nx+1).reshape(1, ny)))
    Kb_rgt = nx-1+nx*arng(ny)
    eb = conc((eb_bot, eb_top, eb_lft, eb_rgt), axis=1)
    Kb = conc((Kb_bot, Kb_top, Kb_lft, Kb_rgt))

    L = arng(1, nx).reshape(1, nx-1)
    Cd = (nx+1)*arng(0, ny).reshape(ny, 1)
    Cf = (nx+1)*arng(1, ny+1).reshape(ny, 1)
    ei_v = np.stack(((L+Cd).ravel(), (L+Cf).ravel()))
    Ki_v = arng(nx*ny).reshape(ny, nx)[:, :(nx-1)].ravel()
    Li_v = arng(nx*ny).reshape(ny, nx)[:, 1:nx].ravel()

    L = (nx+1)*arng(1, ny).reshape(1, ny-1)
    Cd = arng(0, nx).reshape(nx, 1)
    Cf = arng(1, nx+1).reshape(nx, 1)
    ei_h = np.stack(((L+Cd).ravel(), (L+Cf).ravel()))
    Ki_h = arng(nx*ny).reshape(ny, nx)[:(ny-1), :].ravel('F')
    Li_h = arng(nx*ny).reshape(ny, nx)[1:ny, :].ravel('F')

    ei = conc((ei_v, ei_h), axis=1)
    Ki = conc((Ki_v, Ki_h))
    Li = conc((Li_v, Li_h))

    return Mesh(cent, v, ei, eb, Ki, Li, Kb)


def readmesh(s):
    """readmesh(s)
    Read a .msh file to create the associated edge structure.

    At the moment, the mesh must consist only of triangles. The function then
    provides the orthocenters of each triangle to serve as a volume center.
    """
    enc_edges = {}

    def treat_edge(s1, s2):
        e = (min(s1, s2), max(s1, s2))
        if e in enc_edges.keys():
            ei_0.append(s1)
            ei_1.append(s2)
            Ki.append(enc_edges[e])
            Li.append(i)
            del enc_edges[e]
        else:
            enc_edges[e] = i
    prog = 0
    try:
        with open(s) as f:
            fl = f.readline().strip().split()
            prog += 1
            nv = int(fl[0])
            nvol = int(fl[1])
            nvb = int(fl[2])
            cdic = np.zeros((nvol, 2))
            vdic = np.zeros((nv, 2))
            ei_0 = []
            ei_1 = []
            Ki = []
            Li = []
            eb = np.zeros((2, nvb), dtype='int64')
            Kb = np.zeros(nvb, dtype='int64')
            prog += 1
            for i in range(nv):
                line = f.readline().strip().split()
                vdic[i, :] = np.array([float(line[0]), float(line[1])])
            prog += 1
            for i in range(nvol):
                line = f.readline().strip().split()
                if int(line[0]) != 3:
                    raise UnstructuredError(s)
            prog += 1
            curredge = 0
            for i in range(nvol):
                line = f.readline().strip().split()
                voli = [int(s)-1 for s in line[0:-1]]
                cdic[i, :] = orthocenter(vdic[voli[0]], vdic[voli[1]],
                                         vdic[voli[2]])
                s1 = voli.pop()
                s1_0 = s1
                while voli:
                    s2 = voli.pop()
                    treat_edge(s1, s2)
                    s1 = s2
                treat_edge(s1, s1_0)
            ei = np.array([ei_0, ei_1])
            prog += 1
            curredge = 0
            for t, k in enc_edges.items():
                eb[:, curredge] = np.array([t[0], t[1]])
                Kb[curredge] = k
                curredge += 1
            return Mesh(cdic, vdic, ei, eb, np.array(Ki), np.array(Li), Kb)
    except UnstructuredError:
        print(UnstructuredError(s))
    except Exception:
        raise FormatError(s, prog)


def plotdiscrete(E, f, **data):
    """plot(E, f, **data)
    Plot a discrete function f on the mesh with edge structure E.

    The function will draw on the current figure. It will automatically add a
    colorbar at its right.

    Additional arguments can be passed to plot with a keyword framework.
    Current supported keywords are:
    scale : set the scale for the colormap. If scale is absent or the chain
        'free', then the colormap will be streched over the minimum and maximum
        values of f. Otherwise, scale must equal a couple (min, max) and the
        colormap will be stretched over theses values. Note that in this case,
        values of f lower thant min or larger than max will be saturated to the
        corresponding value in the scale.
    """
    fig = plt.gcf()
    mplot = fig.add_axes([0.125, 0.1, 0.8, 0.775])
    mplot.set_xlim((1.1*E.mx-0.1*E.Mx, -0.1*E.mx+1.1*E.Mx))
    mplot.set_ylim((1.1*E.my-0.1*E.My, -0.1*E.my+1.1*E.My))
    mplot.set_aspect('equal')
    mplot.set_anchor('W')
    leg = fig.add_axes([0.85, 0.1, 0.17, 0.775])
#    leg.set_aspect('equal')
    legd = 10
    leg.set(aspect='equal', ylim=(-10, legd*5), xlim = (0, 14))
    leg.set_axis_off()
    leg.set_anchor('W')
    if callable(f):
        val = f(E.centres)
    else:
        val = f
    m = np.min(val)
    M = np.max(val)
    if 'scale' not in data.keys() or data['scale'] == 'free':
        if m == M:
            nf = val/M
        else:
            nf = (val-m)/(M-m)
    else:
        vinf, vsup = data['scale']
        if vinf > m:
            wng.warn('Plot scale cannot handle low values of the function')
        if vsup < M:
            wng.warn('Plot scale cannot handle high values of the function')
        val[val < vinf] = vinf
        val[val > vsup] = vsup
        nf = (val-vinf)/(vsup-vinf)
    paint = plt.cm.get_cmap()

    indiamK = [np.array([E.centres[E.Kin[i]],
                        E.vert[E.inner[0, i]],
                        E.vert[E.inner[1, i]]]) for i in range(E.nin)]
    colK = [paint(nf[E.Kin[i]]) for i in range(E.nin)]
    indiamL = [np.array([E.centres[E.Lin[i]],
                        E.vert[E.inner[0, i]],
                        E.vert[E.inner[1, i]]]) for i in range(E.nin)]
    colL = [paint(nf[E.Lin[i]]) for i in range(E.nin)]
    bndiamK = [np.array([E.centres[E.Kbnd[i]],
                        E.vert[E.bnd[0, i]],
                        E.vert[E.bnd[1, i]]]) for i in range(E.nbnd)]
    colbnd = [paint(nf[E.Kbnd[i]]) for i in range(E.nbnd)]

    P = coll.PolyCollection(indiamK+indiamL+bndiamK,
                            color=colK+colL+colbnd)
    mplot.add_collection(P)
    fig.sca(leg)
    wl, hl = 5, 5
    for i in range(legd):
        plt.fill([0, wl, wl, 0], [i*hl, i*hl, (i+1)*hl, (i+1)*hl],
                 color=paint(i/legd))
        plt.text(wl+0.1, (i+1/3)*hl,
                 "{0:.2e}".format(val[np.argsort(val)[int(i*len(val)/legd)]]))
        leg.set_aspect('equal')
        
# Nouveau
def surf(E, f, ax=None):
    if ax is None:
        ax=plt.gca()
    line = ax.plot_trisurf(E.centres[:, 0], E.centres[:, 1], f)
    return line,




