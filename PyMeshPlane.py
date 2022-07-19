from fnmatch import translate
import math
from operator import index
from tkinter import N
from turtle import ht
import pymesh
import urllib.request
import numpy as np
from scipy.fft import idst
from scipy.misc import face
from scipy.spatial.transform import Rotation as R
from enum import Enum
import abc
import os
import urllib

# Vickers Machine Gun https://www.thingiverse.com/thing:4624542
# Light MG https://www.thingiverse.com/thing:5179363
# 3 Blade Propeller https://www.thingiverse.com/thing:590502/files
# 2 and 4 Blade Propeller https://www.thingiverse.com/thing:2419270/files
# Rotary Engine (also for radial) https://www.thingiverse.com/thing:4303197
# Inline Engine https://www.thingiverse.com/thing:4661707


# docker run -it --mount src="$(pwd)",target=/script,type=bind pymesh/pymesh

debugCockpit = False
debugEngine = False


def transform_pts(vertices, rotation=None, scale=None, translation=None, about_pt=[0, 0, 0]):
    """
    rotate then scale then translate
    Parameters
    vertices (3xfloat): the points to transform
    rotation (3xfloat): yaw pitch roll in degrees. None is no rotation
    translation (3xfloat): how much to shift after rotation. For our purposes it's [Right, Up, Back]. None is no translation
    about_pt (3xfloat, None): The location to rotate about.  [0,0,0] is origin, None is center of object vertices
    """
    vertices = vertices.copy()
    if rotation is not None or scale is not None:
        if any(about_pt):
            vertices -= np.array(about_pt)
        #       roll         yaw          pitch  # zyx?
        # rotation = [-rotation[2], -rotation[0], rotation[1]]
        # [pitch, yaw, roll]
        if rotation is not None:
            rotation = [-rotation[0], rotation[1], -rotation[2]]
            r = R.from_euler('YXZ', rotation, degrees=True)
            vertices = r.apply(vertices)
        if scale is not None:
            vertices *= scale
        if any(about_pt):
            vertices += np.array(about_pt)

    if translation is not None:
        vertices += np.array(translation)

    return vertices


def transform(mesh, rotation=None, scale=None, translation=None, about_pt=[0, 0, 0]):
    """
    rotate then scale then translate
    Parameters
    mesh (mesh): the mesh to transform
    rotation (3xfloat): yaw pitch roll in degrees. None is no rotation
    translation (3xfloat): how much to shift after rotation. For our purposes it's [Right, Up, Back]. None is no translation
    about_pt (3xfloat, None): The location to rotate about.  [0,0,0] is origin, None is center of object vertices
    """
    vertices = mesh.vertices
    faces = mesh.faces

    if about_pt is None:
        about_pt = (mesh.bbox[0]+mesh.bbox[1])/2

    vertices = transform_pts(vertices, rotation, scale, translation, about_pt)

    return pymesh.meshio.form_mesh(vertices, faces)


def parse_airfoil(afurl):
    file = afurl.rpartition('=')[2]
    fspath = '/script/airfoils/'+file+'.dat'
    if os.path.exists(fspath):
        with open(fspath, 'r') as f:
            afdat = f.read()
    else:
        afdat = urllib.request.urlopen(afurl).read().decode('ascii')
        with open(fspath, 'w') as f:
            f.write(afdat)

    aflines = afdat.split('\n')
    airfoil = []
    for pt in aflines[2:]:
        nums = pt.strip().split(None)
        if len(nums) > 1:
            airfoil.append([0, float(nums[1]), float(nums[0])])
    airfoil[0:len(airfoil)//2] = sorted(airfoil[0:len(airfoil)//2],
                                        key=lambda e: e[2])
    airfoil[len(airfoil)//2:] = sorted(airfoil[len(airfoil)//2:], reverse=True,
                                       key=lambda e: e[2])
    airfoil = np.array(airfoil)
    airfoil = airfoil[np.sort(
        np.unique(airfoil, axis=0, return_index=True)[1])]
    airfoil = np.array(airfoil)
    return airfoil


airfoil = parse_airfoil(
    'http://airfoiltools.com/airfoil/lednicerdatfile?airfoil=raf15-il')
symm_airfoil = parse_airfoil(
    'http://airfoiltools.com/airfoil/lednicerdatfile?airfoil=n0009sm-il')
ellipse_airfoil = np.array([[0, -0.075*math.sin(t/64*math.pi - math.pi), 0.5*math.cos(t/64*math.pi - math.pi)+0.5, ]
                           for t in range(128)])


class StackType(Enum):
    SHAPE = 0
    SECTION = 1
    ENGINE = 2
    WINGBANK = 3
    TAILSECTION = 4
    TAIL = 5
    PROPELLER = 6


class EngineType(Enum):
    ROTARY = 'rotary'
    INLINE = 'inline'


class EngineMounting(Enum):
    TRADITIONAL = 1
    CENTER = 2
    POD = 3
    EXPOSED = 4


class EngineDirection(Enum):
    FORWARD = 1
    BACKWARD = 2


class Skin(Enum):
    CLOTH = 1
    NAKED = 2


class Cockpit(Enum):
    NONE = 0
    OPEN = 1
    WINDSCREEN = 2
    NARROW = 3
    BUBBLE = 4


class Layout(Enum):
    NORMAL = 1
    CANARD = 2
    TANDEM = 3
    FARMAN = 4


class StabilizerType(Enum):
    NORMAL = 1
    CANARD_OUTBOARD = 2
    TANDEM_OUTBOARD = 3
    TTAIL = 4
    VTAIL = 5


class Stagger(Enum):
    XPOSITIVE = 2
    POSITIVE = 1
    UNSTAGGERED = 0
    NEGATIVE = -1
    XNEGATIVE = -2


class WingHeight(Enum):
    PARASOL6 = 7
    PARASOL5 = 6
    PARASOL4 = 5
    PARASOL3 = 4
    PARASOL2 = 3
    PARASOL = 2
    SHOULDER = 0.9
    MID = 0
    LOW = -0.9
    GEAR = -2


class PropellerType(Enum):
    TWOBLADE = 2
    THREEBLADE = 3
    FOURBLADE = 4


class FrameShape(Enum):
    CYLINDER = 1
    BOX = 2
    MIXED = 3


class StrutType(Enum):
    NONE = None
    PARALLEL = 0
    N = 1
    V = 2
    I = 3
    W = 4
    SINGLE = 5
    STAR = 6
    ROOT = 7
    TRUSSV = 8
    TRUSSH = 9


class GearType(Enum):
    NONE = 0
    WHEEL = 1
    SKID = 2
    FLOATS = 3


class TipShape(Enum):
    SQUARE = 0
    ROUNDED = 1
    CIRCLE = 2
    SFRONT = 3


skin_thickness = 0.02
frame_types = {FrameShape.CYLINDER: np.array([
    [1., 0., 0.],
    [0.96592583, 0.25881905, 0.],
    [0.8660254, 0.5, 0.],
    [0.70710678, 0.70710678, 0.],
    [0.5, 0.8660254, 0.],
    [0.25881905, 0.96592583, 0.],
    [6.123234e-17, 1.000000e+00, 0.000000e+00],
    [-0.25881905, 0.96592583, 0.],
    [-0.5, 0.8660254, 0.],
    [-0.70710678, 0.70710678, 0.],
    [-0.8660254, 0.5, 0.],
    [-0.96592583, 0.25881905, 0.],
    [-1.0000000e+00, 1.2246468e-16, 0.0000000e+00],
    [-0.96592583, -0.25881905, 0.],
    [-0.8660254, -0.5, 0.],
    [-0.70710678, -0.70710678, 0.],
    [-0.5, -0.8660254, 0.],
    [-0.25881905, -0.96592583, 0.],
    [-1.8369702e-16, -1.0000000e+00, 0.0000000e+00],
    [0.25881905, -0.96592583, 0.],
    [0.5, -0.8660254, 0.],
    [0.70710678, -0.70710678, 0.],
    [0.8660254, -0.5, 0.],
    [0.96592583, -0.25881905, 0.]
]),
    FrameShape.BOX: np.array([
        [0.70710678, 0.70710678, 0.],
        [0.5, 0.70710678, 0.],
        [0.25881905, 0.70710678, 0.],
        [6.123234e-17, 0.70710678, 0.000000e+00],
        [-0.25881905, 0.70710678, 0.],
        [-0.5, 0.70710678, 0.],
        [-0.70710678, 0.70710678, 0.],
        [-0.70710678, -0.70710678, 0.],
        [0.70710678, -0.70710678, 0.]
    ]),
    FrameShape.MIXED: np.array([
        [0.70710678, 0.70710678, 0.],
        [0.5, 0.8660254, 0.],
        [0.25881905, 0.96592583, 0.],
        [6.123234e-17, 1.000000e+00, 0.000000e+00],
        [-0.25881905, 0.96592583, 0.],
        [-0.5, 0.8660254, 0.],
        [-0.70710678, 0.70710678, 0.],
        [-0.70710678, -0.70710678, 0.],
        [0.70710678, -0.70710678, 0.]
    ])}
frame_edges = {
    (FrameShape.CYLINDER, FrameShape.CYLINDER, False): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 0],
        # Stringers
        [0, 24], [1, 25], [2, 26], [3, 27], [4, 28], [5, 29], [6, 30], [7, 31], [8, 32], [9, 33], [10, 34], [11, 35], [
            12, 36], [13, 37], [14, 38], [15, 39], [16, 40], [17, 41], [18, 42], [19, 43], [20, 44], [21, 45], [22, 46], [23, 47],
        # Rear Spars
        [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [
            35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 24]
    ], dtype=np.int32), np.ones([72])*skin_thickness),
    (FrameShape.CYLINDER, FrameShape.CYLINDER, True): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 0],
        # Stringers
        [0, 24], [1, 25], [2, 26], [3, 27], [9, 33], [10, 34], [11, 35], [
            12, 36], [13, 37], [14, 38], [15, 39], [16, 40], [17, 41], [18, 42], [19, 43], [20, 44], [21, 45], [22, 46], [23, 47],
        # Rear Spars
        [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [
            35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 24]
    ], dtype=np.int32), np.ones([67])*skin_thickness),
    (FrameShape.BOX, FrameShape.BOX, False): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0],
        # Longerons
        [0, 9], [6, 15], [7, 16], [8, 17],
        # Rear Spars
        [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 9],
    ], dtype=np.int32), np.ones([22])*2*skin_thickness),
    (FrameShape.BOX, FrameShape.BOX, True): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0],
        # Longerons
        [0, 9], [6, 15], [7, 16], [8, 17],
        # Rear Spars
        [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 9],
    ], dtype=np.int32), np.ones([22])*2*skin_thickness),
    (FrameShape.MIXED, FrameShape.MIXED, False): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0],
        # Longerons
        [0, 9], [6, 15], [7, 16], [8, 17],
        # Rear Spars
        [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 9],
        # Stringers
        [1, 10], [2, 11], [3, 12], [4, 13], [5, 14],
    ], dtype=np.int32), np.hstack((np.ones([22])*2*skin_thickness, np.ones([5])*skin_thickness))),
    (FrameShape.MIXED, FrameShape.MIXED, True): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0],
        # Longerons
        [0, 9], [6, 15], [7, 16], [8, 17],
        # Rear Spars
        [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 9],
    ], dtype=np.int32), np.hstack((np.ones([22])*2*skin_thickness))),
    (FrameShape.CYLINDER, FrameShape.BOX, False): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 0],
        # Stringers
        [0, 24], [1, 24], [2, 24], [3, 24], [4, 24], [5, 24], [6, 24],
        [6, 30], [7, 30], [8, 30], [9, 30], [10, 30], [11, 30], [12, 30],
        [12, 31], [13, 31], [14, 31], [15, 31], [16, 31], [17, 31], [18, 31],
        [18, 32], [19, 32], [20, 32], [21, 32], [22, 32], [23, 32], [0, 32],
        # Longerons
        [3, 24], [9, 30], [15, 31], [21, 32],
        # Rear Spars
        [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [
            29, 30], [30, 31], [31, 32], [32, 24],
    ], dtype=np.int32), np.hstack((np.ones([52])*skin_thickness, np.ones([13])*2*skin_thickness))),
    (FrameShape.CYLINDER, FrameShape.BOX, True): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 0],
        # Stringers
        [0, 24], [1, 24], [2, 24], [3, 24], [
            9, 30], [10, 30], [11, 30], [12, 30],
        [12, 31], [13, 31], [14, 31], [15, 31], [16, 31], [17, 31], [18, 31],
        [18, 32], [19, 32], [20, 32], [21, 32], [22, 32], [23, 32], [0, 32],
        # Longerons
        [3, 24], [9, 30], [15, 31], [21, 32],
        # Rear Spars
        [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [
            29, 30], [30, 31], [31, 32], [32, 24],
    ], dtype=np.int32), np.hstack((np.ones([46])*skin_thickness, np.ones([13])*2*skin_thickness))),
    (FrameShape.CYLINDER, FrameShape.MIXED, False): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 0],
        # Stringers
        [0, 24], [1, 24], [2, 24], [3, 24],
        [4, 25], [5, 26], [6, 27], [7, 28], [8, 29],
        [10, 30], [11, 30], [12, 30],
        [12, 31], [13, 31], [14, 31], [15, 31], [16, 31], [17, 31], [18, 31],
        [18, 32], [19, 32], [20, 32], [21, 32], [22, 32], [23, 32], [0, 32],
        # Rear Spars (narrow)
        [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31],
        # Rear Spars (wide)
        [24, 25], [31, 32], [32, 24],
        # Longerons
        [3, 24], [9, 30], [15, 31], [21, 32],
    ], dtype=np.int32), np.hstack((np.ones([56])*skin_thickness, np.ones([7])*2*skin_thickness))),
    (FrameShape.CYLINDER, FrameShape.MIXED, True): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 0],
        # Stringers
        [0, 24], [1, 24], [2, 24], [3, 24],
        [10, 30], [11, 30], [12, 30],
        [12, 31], [13, 31], [14, 31], [15, 31], [16, 31], [17, 31], [18, 31],
        [18, 32], [19, 32], [20, 32], [21, 32], [22, 32], [23, 32], [0, 32],
        # Rear Spars (narrow)
        [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31],
        # Rear Spars (wide)
        [24, 25], [31, 32], [32, 24],
        # Longerons
        [3, 24], [9, 30], [15, 31], [21, 32],
    ], dtype=np.int32), np.hstack((np.ones([51])*skin_thickness, np.ones([7])*2*skin_thickness))),
    (FrameShape.BOX, FrameShape.MIXED, False): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0],
        # Longerons
        [0, 9], [6, 15], [7, 16], [8, 17],
        # Rear Spars
        [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 9],
        # Stringers
        [1, 10], [2, 11], [3, 12], [4, 13], [5, 14],
    ], dtype=np.int32), np.hstack((np.ones([22])*2*skin_thickness, np.ones([5])*skin_thickness))),
    (FrameShape.BOX, FrameShape.MIXED, True): (np.array([
        # Front Spars
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0],
        # Longerons
        [0, 9], [6, 15], [7, 16], [8, 17],
        # Rear Spars
        [9, 10], [10, 11], [11, 12], [12, 13], [
            13, 14], [14, 15], [15, 16], [16, 17], [17, 9],
    ], dtype=np.int32), np.hstack((np.ones([22])*2*skin_thickness))),
}
wires = (np.array([
    [0.70710678, 0.70710678, 0.],
    [-0.70710678, 0.70710678, 0.],
    [-0.70710678, -0.70710678, 0.],
    [0.70710678, -0.70710678, 0.]
]),
    np.array([
        [0, 5], [1, 4], [1, 6], [2, 7], [3, 4], [0, 7], [2, 5], [3, 6]
    ], dtype=np.int32), 0.25*skin_thickness)


def inflateAndMerge(vertices, edges, thickness, profile_order):
    frame_net = pymesh.WireNetwork.create_from_data(vertices, edges)
    frame_net.add_attribute("vertex_min_angle")
    frame_net.set_attribute("vertex_min_angle", np.ones(
        [vertices.shape[0]])*np.deg2rad(90))
    inflator = pymesh.wires.Inflator(frame_net)
    inflator.set_profile(profile_order)
    inflator.set_refinement(0)
    inflator.inflate(thickness, per_vertex_thickness=False,
                     allow_self_intersection=True)

    mesh = mergeCoPlanar(inflator.mesh, 1.0e-12)
    return mesh


def isCoPlanar(normala, normalb, tol):
    # def clamp(n, minn, maxn): return max(min(maxn, n), minn)
    # phi = math.acos(clamp(np.vdot(normala, normalb) /
    #                 (np.linalg.norm(normala)*np.linalg.norm(normalb)), -1, 1))
    diff = normala-normalb
    return (diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]) < tol*tol


def mergeCoPlanar(mesh, tol=1.0e-6):
    mesh.enable_connectivity()
    mesh.add_attribute("face_normal")
    normals = np.reshape(
        np.array(mesh.get_attribute("face_normal")), mesh.faces.shape)

    reverse_faces = [None]*mesh.faces.shape[0]
    face_mapping = []
    num_big_faces = 0
    for idx in range(mesh.faces.shape[0]):
        adjacency = mesh.get_face_adjacent_faces(idx)
        normal_self = normals[idx]
        for adj in adjacency:
            if idx > adj:
                normal_adj = normals[adj]
                if isCoPlanar(normal_adj, normal_self, tol):
                    if reverse_faces[idx] is None:
                        reverse_faces[idx] = reverse_faces[adj]
                    else:
                        mn = min(reverse_faces[idx], reverse_faces[adj])
                        face_mapping[max(
                            reverse_faces[idx], reverse_faces[adj])] = face_mapping[mn]
                        reverse_faces[idx] = face_mapping[mn]
        if reverse_faces[idx] is None:
            reverse_faces[idx] = num_big_faces
            face_mapping.append(num_big_faces)
            num_big_faces += 1
        for i in range(len(face_mapping)):
            face_mapping[i] = face_mapping[face_mapping[i]]
    for i in range(len(reverse_faces)):
        reverse_faces[i] = face_mapping[reverse_faces[i]]

    # Make a list of all the vertex indexes that are on three or more faces
    num_vertices = mesh.vertices.shape[0]
    vertices_to_keep = []
    for i in range(num_big_faces):
        vertices_to_keep.append([])

    # Every vertex is in at least one face
    for v in range(num_vertices):
        # Get the (small) faces the vertex touches
        adj_fidx = mesh.get_vertex_adjacent_faces(v)
        # Count how many unique (big) faces it touches
        face_set = set([])
        for fidx in adj_fidx:
            face_set.add(reverse_faces[fidx])
        # Keep the ones that touch 3 or more (meaning at a joint, not an edge between two faces or internal to one face)
        if len(face_set) > 2:
            for f in face_set:
                vertices_to_keep[f].append(v)

    final_meshes = []
    # Triangularize each face using the external vertices
    for vs in vertices_to_keep:
        if len(vs) < 3:
            continue
        # Get the actual vertex coordinates for these vertex indexes
        vertices = mesh.vertices[vs, :]
        # And triangularize
        tri = pymesh.triangle()
        tri.points = vertices
        tri.max_area = 100
        tri.split_boundary = False
        tri.verbosity = 0
        tri.run()  # Execute triangle.
        final_meshes.append(tri.mesh)  # output triangulation.

    return pymesh.merge_meshes(final_meshes)


def make_section_vertices(front_type, rear_type, front_size, rear_size, length):
    front_shape = frame_types[front_type] * front_size
    rear_shape = frame_types[rear_type] * rear_size
    rear_shape[:, 2] += length
    if front_type.value <= rear_type.value:
        vertices = np.vstack((front_shape, rear_shape))
    else:
        vertices = np.vstack((rear_shape, front_shape))
    return vertices


def make_wire_vertices(front_size, rear_size, length):
    front_shape = wires[0] * front_size
    rear_shape = wires[0] * rear_size
    rear_shape[:, 2] += length
    vertices = np.vstack((front_shape, rear_shape))
    return vertices


def make_shell(front_type, rear_type, front_size, rear_size, length):
    vertices = make_section_vertices(
        front_type, rear_type, front_size, rear_size, length)
    tri = pymesh.tetgen()
    tri.points = vertices
    tri.verbosity = 0
    tri.run()
    return tri.mesh


def make_skin(front_type, rear_type, front_size, rear_size, length, opentop):
    outer = make_shell(front_type, rear_type, front_size +
                       skin_thickness, rear_size+skin_thickness, length)
    inner = make_shell(front_type, rear_type, front_size, rear_size, length)
    skin = pymesh.boolean(outer, inner, 'difference')
    if opentop:
        vertices = wires[0][0:2, :] * [1, 0, 1]
        vertices[0, 0] -= skin_thickness/front_size
        vertices[1, 0] += skin_thickness/front_size
        vertices = np.vstack(
            [front_size*vertices, rear_size*vertices+[0, 0, length]])
        vertices = np.vstack([vertices, vertices+[0, 100, 0]])
        tri = pymesh.tetgen()
        tri.points = vertices
        tri.verbosity = 0
        tri.run()
        skin = pymesh.boolean(skin, tri.mesh, 'difference')
    return skin


def make_frame(front_type, rear_type, front_size, rear_size, length, opentop):
    vertices = make_section_vertices(
        front_type, rear_type, front_size-skin_thickness, rear_size-skin_thickness, length)

    if front_type.value <= rear_type.value:
        edges, thickness = frame_edges[(front_type, rear_type, opentop)]
    else:
        edges, thickness = frame_edges[(rear_type, front_type, opentop)]

    frame_mesh = inflateAndMerge(vertices, edges, thickness, 8)

    wire_vertexes = make_wire_vertices(
        front_size-skin_thickness, rear_size-skin_thickness, length)
    wire_edges = wires[1]
    if opentop:
        wire_edges = wire_edges[2:]
    wire_mesh = inflateAndMerge(wire_vertexes, wire_edges, wires[2], 3)
    frame_mesh = pymesh.merge_meshes([frame_mesh, wire_mesh])

    return frame_mesh


def make_both(front_type, rear_type, front_size, rear_size, length, opentop):
    skin = make_skin(front_type, rear_type, front_size,
                     rear_size, length, opentop)
    frame = make_frame(front_type, rear_type, front_size,
                       rear_size, length, opentop)
    return pymesh.merge_meshes([skin, frame])


frame_length = 1.25
frame_width = 0.5
tail_width = 0.25

# List of wing banks, each is a list of wings
# A wing is a tuple of Height, Area, Span, Dihedral, TipShape
# Fictional Tails to adjust tail area
# Shape is: (FrameShape,)
# Propeller is: (PropellerType,)
# Engine is: (EngineType, EngineMounting, EngineDirection)
# Section is: (FrameShape, Skin, Cockpit, oldsize (float), newsize (float))
# TailSection is: (num sections (int), Layout, Skin)
# WingBank is: ([(WingHeight, Area, Span, Dihedral, Sweep)],Stagger, StrutType (Cabane), [StrutTypes], GearType)
# Tail is: (StabilizerType, nHStab (int), nVstab (int), sizeFactor(float))
# Slice is: [Other Parts]
# Stack is: [Slices]
# Plane is: ([Stacks], [offsets (3x1 float)])


class Part(abc.ABC):
    @abc.abstractclassmethod
    def build(self, shape, size):
        pass
    # Mesh, Length, Shape, Size

    def wing_stats(self):
        return 0, 0

    def set_wing_stats(self, area, span):
        pass


class Shape(Part):
    def __init__(self, shape=None, size=None) -> None:
        super().__init__()
        self.shape = shape
        self.size = size

    def build(self, shape, size):
        if self.shape is not None:
            shape = self.shape
        if self.size is not None:
            size = self.size
        return None, None, shape, size


class Propeller(Part):
    propeller_types = {PropellerType.TWOBLADE: '/script/objects/2BladeProp.obj',
                       PropellerType.THREEBLADE: '/script/objects/3BladeProp.obj',
                       PropellerType.FOURBLADE: '/script/objects/2BladeProp.obj', }

    def __init__(self, prop_type=PropellerType.TWOBLADE) -> None:
        super().__init__()
        self.prop_type = prop_type

    def build(self, shape, _):
        propeller_file = self.propeller_types[self.prop_type]
        propeller_mesh = pymesh.meshio.load_mesh(propeller_file)
        if self.prop_type == PropellerType.FOURBLADE:
            propeller_mesh = pymesh.merge_meshes(
                [propeller_mesh, transform(propeller_mesh, rotation=[0, 0, 90])])
        prop_length = propeller_mesh.bbox[1][2]
        return propeller_mesh, prop_length, shape, None


class Engine(Part):
    engine_types = {EngineType.ROTARY: (0.33, 0.4, '/script/objects/Rotary.obj', None),
                    EngineType.INLINE: (1.4, 0.3, '/script/objects/Inline.obj', None)}

    def __init__(self, etype=EngineType.ROTARY, mounting=EngineMounting.TRADITIONAL, direction=EngineDirection.FORWARD, rear_firewall=False, front_firewall=False, new_size=frame_width) -> None:
        super().__init__()
        self.etype = etype
        self.mounting = mounting
        self.direction = direction
        self.rear_firewall = rear_firewall
        self.front_firewall = front_firewall
        self.new_size = new_size

    def build(self, shape, size):
        l_engine, r_engine, engine_file, engine_mesh = Engine.engine_types[self.etype]
        if self.direction == EngineDirection.FORWARD:
            front = size
            back = self.new_size
        else:
            front = self.new_size
            back = size
        opentop = self.etype == EngineType.INLINE
        if self.mounting == EngineMounting.TRADITIONAL:
            section = make_section(
                l_engine, front, back, shape, shape, firewall_rear=self.rear_firewall, firewall_front=self.front_firewall, opentop=opentop)
        elif self.mounting == EngineMounting.EXPOSED:
            section = make_section(
                l_engine, front, back, shape, shape, firewall_rear=self.rear_firewall, firewall_front=self.front_firewall, skin=Skin.NAKED, opentop=opentop)
        else:
            section = make_section(
                l_engine, front, back, shape, shape, firewall_front=self.front_firewall, firewall_rear=self.rear_firewall, opentop=opentop)

        if engine_mesh is None:
            if debugEngine:
                engine_mesh = pymesh.generate_cylinder(
                    [0, 0, 0], [0, 0, l_engine], 0.8*r_engine, 0.8*r_engine, 128)
            else:
                engine_mesh = pymesh.meshio.load_mesh(engine_file)
            Engine.engine_types[self.etype] = (
                l_engine, r_engine, engine_file, engine_mesh)
        else:
            pass

        engine_block = pymesh.merge_meshes([engine_mesh, section])

        if self.direction == EngineDirection.BACKWARD:
            engine_block = transform(engine_block, rotation=[
                180, 0, 0], about_pt=None)
        length = engine_block.bbox[1][2]
        return engine_block, length, shape, back


class Section(Part):
    def __init__(self, shape=None, skin=Skin.CLOTH, cockpit=Cockpit.NONE, opentop=False, newsize=None) -> None:
        super().__init__()
        self.shape = shape
        self.skin = skin
        self.cockpit = cockpit
        self.opentop = opentop
        self.new_width = newsize

    def build(self, shape, size):
        if self.shape is None:
            newShape = shape
        else:
            newShape = self.shape
        if self.new_width is None:
            new_width = size
        else:
            new_width = self.new_width

        has_cockpit = self.cockpit != Cockpit.NONE
        parts = []
        parts.append(make_section(
            frame_length, size, new_width, shape, newShape, self.skin, opentop=self.opentop, firewall_front=has_cockpit))

        if has_cockpit:
            if debugCockpit:
                seat = pymesh.generate_box_mesh(
                    [-0.33866, -0.331604,  0.006], [0.337532, 0.493043, 1.292496])
            else:
                seat = pymesh.load_mesh('/script/objects/Cockpit4.obj')
            if self.cockpit == Cockpit.WINDSCREEN:
                parts.append(self.create_windscreen(shape))
            elif self.cockpit == Cockpit.NARROW:
                canopy = self.create_narrow(shape, newShape)
                seat = pymesh.boolean(seat, pymesh.generate_box_mesh(
                    canopy.bbox[0], canopy.bbox[1]), 'difference')
                parts.append(canopy)
            parts.append(seat)
        return pymesh.merge_meshes(parts), frame_length, newShape, self.new_width

    def create_windscreen(self, shape):
        vertices = np.zeros([6, 3])
        if shape == FrameShape.CYLINDER:
            vertices[0, :] = frame_types[shape][4, :]
            vertices[1, :] = frame_types[shape][8, :]
        else:
            vertices[0, :] = frame_types[shape][1, :]
            vertices[1, :] = frame_types[shape][5, :]
        vertices[2, :] = [0.5, 1.25, 0.2]
        vertices[3, :] = [-0.5, 1.25, 0.2]
        vertices[4, :] = [0.70710678, 0.70710678, 0.5]
        vertices[5, :] = [-0.70710678, 0.70710678, 0.5]
        vertices *= frame_width
        edges = np.array(
            [[0, 2], [1, 3], [2, 3], [2, 4], [3, 5]], dtype=np.int32)
        return inflateAndMerge(vertices, edges, skin_thickness, 4)

    def create_narrow(self, shape, newshape):
        vertices = np.zeros([12, 3])
        if shape == FrameShape.CYLINDER:
            vertices[0, :] = frame_types[shape][5, :]
            vertices[1, :] = frame_types[shape][7, :]
        elif shape == FrameShape.BOX or shape == FrameShape.MIXED:
            vertices[0, :] = frame_types[shape][2, :]
            vertices[1, :] = frame_types[shape][4, :]
        else:
            raise "Not Implemented Yet"
        vertices[2, :] = [0.25881905, 1.3, 0.5]
        vertices[3, :] = [-0.25881905, 1.3, 0.5]
        vertices[4, :] = [0.70710678, 0.70710678, 0.5]
        vertices[5, :] = [-0.70710678, 0.70710678, 0.5]
        vertices[6, :] = [0.25881905, 1.3, 1.25]
        vertices[7, :] = [-0.25881905, 1.3, 1.25]
        if newshape == FrameShape.CYLINDER:
            vertices[8, :] = frame_types[newshape][5, :] + \
                np.array([0, 0, 1.25])
            vertices[9, :] = frame_types[newshape][7, :] + \
                np.array([0, 0, 1.25])
        elif newshape == FrameShape.BOX or shape == FrameShape.MIXED:
            vertices[8, :] = frame_types[newshape][2, :] + \
                np.array([0, 0, 1.25])
            vertices[9, :] = frame_types[newshape][4, :] + \
                np.array([0, 0, 1.25])
        else:
            raise "Not Implemented Yet"
        vertices[10, :] = [0.70710678, 0.70710678, 1.25]
        vertices[11, :] = [-0.70710678, 0.70710678, 1.25]

        vertices *= np.array([frame_width-skin_thickness,
                              frame_width-skin_thickness, 1])
        edges = np.array(
            [[0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [2, 6], [3, 7], [6, 7], [6, 8], [6, 10], [7, 9], [7, 11]], dtype=np.int32)
        frame_mesh = inflateAndMerge(vertices, edges, skin_thickness, 4)

        scale_arr = np.array(
            [frame_width-skin_thickness, frame_width-skin_thickness, 1])
        tri = pymesh.tetgen()
        tri.points = vertices
        tri.verbosity = 0
        tri.run()
        glass = tri.mesh
        tri.points = np.vstack((vertices[0, :],
                                vertices[4, :], np.array([[0.70710678, 0.70710678, 0],
                                                          [0, 0.7, 0]])*scale_arr))
        tri.verbosity = 0
        tri.run()
        glassp = tri.mesh
        tri.points = np.vstack((vertices[1, :],
                                vertices[5, :],
                                np.array([[-0.70710678, 0.70710678, 0],
                                          [0, 0.7, 0]])*scale_arr,))
        tri.verbosity = 0
        tri.run()
        glassn = tri.mesh
        return pymesh.merge_meshes([frame_mesh, glass, glassp, glassn])


class TailSection(Part):
    def __init__(self, tail_sections=1, skin=Skin.CLOTH, rear_size=None, rear_shape=FrameShape.CYLINDER) -> None:
        super().__init__()
        self.tail_sections = tail_sections
        self.skin = skin
        self.final_size = rear_size
        self.final_shape = rear_shape

    def build(self, shape, size):
        if self.final_size is None:
            self.final_size = size
        parts = []
        for i in range(self.tail_sections):
            b = (i+1)*(self.final_size-size) / self.tail_sections+size
            if i == self.tail_sections-1:
                parts.append(Section(skin=self.skin, shape=self.final_shape,
                                     newsize=b))
            else:
                parts.append(Section(skin=self.skin, newsize=b))

        meshes = []
        tail_stack_point = 0
        for part in parts:
            mesh, length, newshape, size = part.build(shape, size)
            if mesh is not None:
                meshes.append(transform(mesh, translation=[
                              0, 0, tail_stack_point]))
            tail_stack_point += length
            if newshape is not None:
                shape = newshape
        mesh = pymesh.merge_meshes(meshes)
        return mesh, tail_stack_point, shape, size


class Wing():
    def __init__(self, stuff, airfoil) -> None:
        height, area, span, dihedral, sweep, tip = stuff
        self.height = height
        self.area = area
        self.span = span
        self.dihedral = dihedral
        if sweep >= 0 and sweep < 80:
            self.sweep = max(sweep, 0.1)
        else:
            self.sweep = sweep
        self.tip = tip
        self.distance_back = 0
        if isinstance(airfoil, tuple):
            af0_size = airfoil[0].shape[0]
            af1_size = airfoil[1].shape[0]
            af0_top_x = airfoil[0][0:af0_size//2+1, 2]
            af0_bottom_x = np.hstack(
                (airfoil[0][0, 2], airfoil[0][:af0_size//2-1:-1, 2]))
            af0_top_y = airfoil[0][0:af0_size//2+1, 1]
            af0_bottom_y = np.hstack(
                (airfoil[0][0, 1], airfoil[0][:af0_size//2-1:-1, 1]))
            af1_top_x = airfoil[1][0:af1_size//2+1, 2]
            af1_bottom_x = np.hstack(
                (airfoil[1][0, 2], airfoil[1][:af1_size//2-1:-1, 2]))
            af1_top_y = airfoil[1][0:af1_size//2+1, 1]
            af1_bottom_y = np.hstack(
                (airfoil[1][0, 1], airfoil[1][:af1_size//2-1:-1, 1]))

            if af0_size > af1_size:
                top_x = af0_top_x
                bottom_x = af0_bottom_y
            else:
                top_x = af1_top_x
                bottom_x = af1_bottom_x

            af0_top_y = np.interp(top_x, af0_top_x, af0_top_y)
            af0_bottom_y = np.interp(bottom_x, af0_bottom_x, af0_bottom_y)
            af1_top_y = np.interp(top_x, af1_top_x, af1_top_y)
            af1_bottom_y = np.interp(bottom_x, af1_bottom_x, af1_bottom_y)
            afn_x = np.hstack((top_x, bottom_x[-2:0:-1])).reshape(-1, 1)
            af0_y = np.hstack(
                (af0_top_y, af0_bottom_y[-2:0:-1])).reshape(-1, 1)
            af1_y = np.hstack(
                (af1_top_y, af1_bottom_y[-2:0:-1])).reshape(-1, 1)

            def interpAirfoil(span_frac):
                new_y = (af1_y - af0_y)*span_frac+af0_y
                temp = np.hstack((np.zeros(afn_x.shape), new_y, afn_x))
                return temp

            self.airfoil = interpAirfoil
        else:
            self.airfoil = lambda span_loc: airfoil

        def NoSize(_):
            raise "No Size Set."
        self.leading_edge = NoSize
        self.chord = NoSize

    def set_size(self, size, start_span):
        height = self.height.value*size/math.sqrt(2)

        def flat_leading_edge(span_loc):
            span_loc = min(span_loc, self.span/2)
            sweep_distance = span_loc*math.tan(np.deg2rad(self.sweep))
            return np.array([span_loc, height+0.25*self.dihedral, sweep_distance+self.distance_back])

        def flat_chord(span_loc):
            span_loc = min(span_loc, self.span/2)
            return self.area/self.span

        if self.sweep < 0:
            def circ_leading_edge(theta):
                theta = math.pi/2-theta
                odist = abs(self.sweep/1000)*frame_length
                idist = odist - self.area/self.span
                span_loc = (self.span/2 - self.area/self.span)*math.cos(theta)
                span_loc = max(span_loc, start_span)
                sweep_distance = -(idist - math.sin(theta)*idist)
                return np.array([span_loc, height+0.25*self.dihedral, sweep_distance+self.distance_back])
            self.circ_leading_edge = circ_leading_edge

            def half_chord(theta):
                rot = theta*180/math.pi
                if self.sweep < 0:
                    rot *= -1
                return self.circ_leading_edge(theta) + transform_pts(np.array([[0, 0, flat_chord(0)/2]]), rotation=[rot, 0, 0])
            self.half_chord = half_chord

        elif self.sweep > 80:
            def circ_leading_edge(theta):
                theta = math.pi/2-theta
                odist = abs(self.sweep/1000)*frame_length
                idist = odist - self.area/self.span
                span_loc = (self.span/2) * math.cos(theta)
                span_loc = max(span_loc, start_span)
                sweep_distance = odist - math.sin(theta)*odist
                return np.array([span_loc, height+0.25*self.dihedral, sweep_distance+self.distance_back])
            self.circ_leading_edge = circ_leading_edge

            def half_chord(theta):
                rot = theta*180/math.pi
                if self.sweep < 0:
                    rot *= -1
                return self.circ_leading_edge(theta) + transform_pts(np.array([[0, 0, flat_chord(0)/2]]), rotation=[rot, 0, 0])
            self.half_chord = half_chord

        if self.tip == TipShape.ROUNDED:
            def leading_edge(span_loc):
                span_loc = min(span_loc, self.span/2)
                le = flat_leading_edge(span_loc)
                c = flat_chord(span_loc)
                start_tip = (self.span/2)-0.45*c
                if span_loc > start_tip:
                    lediff = c/2 - \
                        math.sin(
                            math.acos(min(1, 2*(span_loc-start_tip)/c)))*c/2
                    le += np.array([0, 0, lediff])
                return le

            def chord(span_loc):
                span_loc = min(span_loc, self.span/2)
                c = flat_chord(span_loc)
                start_tip = (self.span/2)-0.45*c
                if span_loc > start_tip:
                    lediff = c/2 - \
                        math.sin(
                            math.acos(min(1, 2*(span_loc-start_tip)/c)))*c/2
                    c -= 2*lediff
                return c
        elif self.tip == TipShape.CIRCLE:
            def leading_edge(span_loc):
                span_loc = min(span_loc, self.span/2)
                le = flat_leading_edge(span_loc)
                c = flat_chord(span_loc)
                start_tip = (self.span/2)-c/2
                if span_loc > start_tip:
                    lediff = c/2 - \
                        math.sin(
                            math.acos(min(1, 2*(span_loc-start_tip)/c)))*c/2
                    le += np.array([0, 0, lediff])
                return le

            def chord(span_loc):
                span_loc = min(span_loc, self.span/2)
                c = flat_chord(span_loc)
                start_tip = (self.span/2)-c/2
                if span_loc > start_tip:
                    lediff = c/2 - \
                        math.sin(
                            math.acos(min(1, 2*(span_loc-start_tip)/c)))*c/2
                    c -= 2*lediff
                return c
        elif self.tip == TipShape.SFRONT:
            def leading_edge(span_loc):
                span_loc = min(span_loc, self.span/2)
                le = flat_leading_edge(span_loc)
                c = flat_chord(span_loc)
                start_tip = (self.span/2)-c
                if span_loc > start_tip:
                    lediff = c-math.sin(math.acos((span_loc-start_tip)/c))*c
                    le += np.array([0, 0, lediff])
                return le

            def chord(span_loc):
                span_loc = min(span_loc, self.span/2)
                c = flat_chord(span_loc)
                start_tip = (self.span/2)-c
                if span_loc > start_tip:
                    lediff = c-math.sin(math.acos((span_loc-start_tip)/c))*c
                    c -= lediff
                return c
        else:
            leading_edge = flat_leading_edge
            chord = flat_chord

        if not hasattr(self, "half_chord"):
            self.half_chord = lambda span_loc: leading_edge(
                span_loc) + [0, 0, chord(span_loc)/2]

        self.leading_edge = leading_edge
        self.chord = chord

    def wing_location(self, span_loc):
        if self.sweep > 0 and self.sweep < 80:
            return self.leading_edge(span_loc), self.chord(span_loc)
        else:
            theta = math.pi/2 - math.acos(span_loc / (self.span/2))
            return self.circ_leading_edge(theta), self.chord(theta)

    def make_wing_skin(self, af_size, mesh_size):
        new_faces = []
        for na, nb in zip(range(af_size), [(x if x < af_size else 0) for x in range(1, af_size+1)]):
            oa = na - mesh_size
            ob = nb - mesh_size
            new_faces.append([oa, na, ob])
            new_faces.append([na, nb, ob])
        return np.array(new_faces)

    def make_wing(self, start_span, steps=128):
        af_size = self.airfoil(0).shape[0]
        end_span = self.span/2

        def airfoil_mesh(span_loc):
            tri = pymesh.triangle()
            tri.points = self.airfoil(span_loc)
            tri.segments = np.array([(x, x+1 if x+1 < af_size else 0)
                                     for x in range(0, af_size)])
            tri.verbosity = 0
            tri.run()
            return tri.mesh
        shape_mesh = airfoil_mesh(0)
        mesh_size = shape_mesh.vertices.shape[0]

        faces = shape_mesh.faces
        new_faces = self.make_wing_skin(af_size, mesh_size)

        if self.sweep >= 0 and self.sweep < 80:
            vertices = transform_pts(shape_mesh.vertices, scale=self.chord(
                start_span), translation=self.leading_edge(start_span))
            offset = mesh_size
            for step in range(1, steps+1):
                span_loc = (end_span-start_span)*step/steps+start_span
                shape_mesh = airfoil_mesh(step/steps)
                new_vertices = transform_pts(
                    shape_mesh.vertices, scale=self.chord(span_loc), translation=self.leading_edge(span_loc))
                vertices = np.vstack((vertices, new_vertices))
                new_faces = self.make_wing_skin(af_size, mesh_size)
                faces = np.vstack((faces, new_faces+offset))
                if step != steps:
                    offset += new_vertices.shape[0]
                mesh_size = new_vertices.shape[0]
            faces = np.vstack((faces, shape_mesh.faces+offset))
        else:
            vertices = transform_pts(shape_mesh.vertices, scale=self.chord(
                start_span), translation=self.circ_leading_edge(0))
            offset = mesh_size
            for step in range(1, steps+1):
                theta = math.pi/2*step/steps
                rot = theta*180/math.pi
                if self.sweep < 0:
                    rot *= -1
                shape_mesh = airfoil_mesh(step/steps)
                new_vertices = transform_pts(
                    shape_mesh.vertices, rotation=[rot, 0, 0], scale=self.chord(math.pi/2-theta), translation=self.circ_leading_edge(theta))
                vertices = np.vstack((vertices, new_vertices))
                new_faces = self.make_wing_skin(af_size, mesh_size)
                faces = np.vstack((faces, new_faces+offset))
                if step != steps:
                    offset += new_vertices.shape[0]
                mesh_size = new_vertices.shape[0]
            faces = np.vstack((faces, shape_mesh.faces+offset))

        right = pymesh.form_mesh(vertices, faces)
        left = pymesh.form_mesh(vertices*np.array([-1, 1, 1]), faces)
        return right, left


class WingBank(Part):
    def __init__(self, wings, stagger=Stagger.UNSTAGGERED, cabane=StrutType.NONE, struts=[], gear=GearType.NONE, wires=False) -> None:
        super().__init__()
        self.wings = wings
        self.wings.sort(key=lambda y: y.height.value, reverse=True)
        self.staggers = []
        self.cabane = cabane
        self.struts = [strut for strut in struts if strut !=
                       StrutType.TRUSSH and strut != StrutType.TRUSSV]
        self.num_trussv = struts.count(StrutType.TRUSSV)
        self.num_trussh = struts.count(StrutType.TRUSSH)
        struts.reverse()
        self.gear = gear
        self.wires = wires

        stagger_dist = 0
        min_stagger = 1000
        for w in self.wings:
            chord = w.area/w.span
            min_stagger = min(min_stagger, stagger_dist)
            self.staggers.append(stagger_dist)
            stagger_dist += 1*chord/4 * stagger.value
        for s in range(len(self.staggers)):
            self.staggers[s] -= min_stagger

        for w, s in zip(self.wings, self.staggers):
            w.distance_back = s

        if len(self.staggers) == 0:
            self.staggers.append(0)
        self.default_wing = Wing(
            ((WingHeight.MID, 1, 1, 0, 0, TipShape.SQUARE)), airfoil)

    def build(self, _, size):
        wing_meshes = []
        self.default_wing.set_size(size, 0)
        for w in self.wings:
            if w.area > 0 and w.span > 0:
                if w.height.value > 1 or w.height.value < -1:
                    start_span = 0
                else:
                    start_span = size/math.sqrt(2)+skin_thickness
                w.set_size(size, start_span)
                r, l = w.make_wing(start_span)
                wing_meshes.append(l)
                wing_meshes.append(r)

        meshes, wire_points = self.build_struts(size)
        if self.wires and len(wire_points) > 2:
            meshes.append(self.build_wires(wire_points))

        if self.num_trussv > 0:
            meshes += self.build_vtruss(size)

        if len(wing_meshes) > 0:
            wing_mesh = pymesh.merge_meshes(wing_meshes)
            meshes.append(wing_mesh)

        return pymesh.merge_meshes(meshes), None, None, None

    def wing_stats(self):
        area = 0
        span = 0
        for w in self.wings:
            area += w.area
            span = max(span, w.span)
        return area, span

    def strut_points(self, size, span_loc_top, span_loc_bot, force_top=None, force_bottom=None):
        topw = self.wings[0] if len(
            self.wings) >= 1 else self.default_wing
        botw = self.wings[-1] if len(
            self.wings) >= 1 else self.default_wing

        location_top, chordt = topw.wing_location(span_loc_top)
        location_bottom, chordb = botw.wing_location(span_loc_bot)

        if force_top is not None:
            location_top = np.array(
                [abs(force_top), force_top, location_bottom[2]])
        if force_bottom is not None:
            location_bottom = np.array(
                [abs(force_bottom), force_bottom, location_top[2]])

        return location_top, location_bottom, chordt, chordb

    def build_struts(self, size):
        _, s = self.wing_stats()
        scount = len(self.struts) + (0 if self.cabane == StrutType.NONE else 1)
        for strut in self.struts:
            if strut == StrutType.STAR:
                scount += 1
        locations = [spl * s / (2*scount+1) - s/2
                     for spl in range(1, 2*scount+1)]
        locations.reverse()
        locations = [l for l in locations if l > 0]

        topw = self.wings[0] if len(
            self.wings) >= 1 else self.default_wing
        botw = self.wings[-1] if len(
            self.wings) > 1 else self.default_wing
        meshes = []
        wire_points = []
        for s in self.struts:
            lt = locations[0]
            ti = 1
            lb = locations[0]
            tb = 1
            while lt > topw.span/2-0.1:
                lt = locations[ti]
                ti += 1
            while lb > botw.span/2-0.1:
                lb = locations[tb]
                tb += 1

            locations = locations[1:]
            le_top, le_bottom, chord_top, chord_bottom = self.strut_points(
                size, lt, lb)

            if s == StrutType.STAR:
                lt2 = locations[0]
                lb2 = locations[0]
                locations = locations[1:]
                le_top2, le_bottom2, c2_top, c2_bottom = self.strut_points(
                    size, lt2, lb2)
            else:
                le_top2 = []
                le_bottom2 = []

            strut = Strut(le_top, le_bottom,
                          chord_top, chord_bottom,
                          le_top2=le_top2, le_bottom2=le_bottom2)
            m, wp = strut.build(s)
            meshes.append(m)
            wire_points += wp

        frame_corner = size/math.sqrt(2)
        center_points = [np.array([frame_corner, frame_corner, 0]), np.array([frame_corner, -frame_corner, 0]), np.array(
            [frame_corner, frame_corner, frame_length]), np.array([frame_corner, -frame_corner, frame_length])]
        if self.cabane != StrutType.NONE or self.gear != GearType.NONE:
            s = self.cabane
            # Upper Cabane
            if topw.height.value > 1 and s != StrutType.NONE:
                lt = locations[0]
                lb = frame_corner
                le_top, le_bottom, chord_top, chord_bottom = self.strut_points(size,
                                                                               lt, lb, force_bottom=frame_corner)

                if s == StrutType.STAR:
                    le_top2, le_bottom2, c2_top, c2_bottom = self.strut_points(size,
                                                                               -lt, -lb, force_bottom=frame_corner)
                else:
                    le_top2 = []
                    le_bottom2 = []

                strut = Strut(le_top, le_bottom,
                              chord_top, chord_top,
                              le_top2=le_top2, le_bottom2=le_bottom2)
                m, wp = strut.build(s)
                meshes.append(m)
                center_points[0] = wp[0]
                center_points[2] = wp[2]
            # Lower Cabane
            if botw.height.value < -1 or self.gear != GearType.NONE:
                if self.gear != GearType.NONE:
                    s = StrutType.V
                    if len(locations) > 0:
                        locations[0] = frame_corner
                    else:
                        locations.append(frame_corner)

                lt = frame_corner
                lb = locations[0]
                le_top, le_bottom, chord_top, chord_bottom = self.strut_points(size,
                                                                               lt, lb, force_top=-frame_corner)
                le_bottom[1] = min(-1, le_bottom[1])

                if s == StrutType.STAR:
                    le_top2, le_bottom2, c2_top, c2_bottom = self.strut_points(size,
                                                                               -lt, -lb, force_top=-frame_corner)
                else:
                    le_top2 = []
                    le_bottom2 = []

                strut = Strut(le_top, le_bottom,
                              chord_bottom, chord_bottom,
                              gear=(self.gear != GearType.NONE),
                              le_top2=le_top2, le_bottom2=le_bottom2)
                m, wp = strut.build(s)
                meshes.append(m)
                center_points[1] = wp[1]
                center_points[3] = wp[3]

                if self.gear == GearType.WHEEL:
                    meshes.append(pymesh.generate_cylinder(le_bottom + np.array([2*skin_thickness, 0, chord_bottom/5]),
                                                           le_bottom + np.array([4*skin_thickness, 0, chord_bottom/5]), 0.5*frame_width, 0.5*frame_width, 64))
                    meshes.append(pymesh.generate_cylinder(np.array([-1, 1, 1])*le_bottom+np.array([-2*skin_thickness, 0, chord_bottom/5]),
                                                           np.array([-1, 1, 1])*le_bottom + np.array([-4*skin_thickness, 0, chord_bottom/5]), 0.5*frame_width, 0.5*frame_width, 64))
                    meshes.append(pymesh.generate_cylinder(
                        le_bottom + np.array([0, 0, chord_bottom/5]), np.array([-1, 1, 1])*le_bottom + np.array([0, 0, chord_bottom/5]), skin_thickness, skin_thickness))
                elif self.gear == GearType.SKID:
                    meshes.append(pymesh.generate_cylinder(
                        le_bottom + np.array([0, 0, chord_bottom/5]), np.array([-1, 1, 1])*le_bottom + np.array([0, 0, chord_bottom/5]), skin_thickness, skin_thickness))
                    meshes.append(pymesh.generate_box_mesh(le_bottom + np.array(
                        [-4*skin_thickness, -skin_thickness, 0]), le_bottom + np.array([4*skin_thickness, 0, chord_bottom])))
                    meshes.append(pymesh.generate_box_mesh(np.array([-1, 1, 1])*le_bottom + np.array(
                        [-4*skin_thickness, -skin_thickness, 0]), np.array([-1, 1, 1])*le_bottom + np.array([4*skin_thickness, 0, chord_bottom])))
                elif self.gear == GearType.FLOATS:
                    meshes.append(pymesh.generate_cylinder(le_bottom + np.array(
                        [0, -5*skin_thickness, 0]), le_bottom + np.array(
                        [0, -5*skin_thickness, frame_length]), 8*skin_thickness, 5*skin_thickness))
                    meshes.append(pymesh.generate_cylinder(np.array([-1, 1, 1])*le_bottom + np.array(
                        [0, -5*skin_thickness, 0]), np.array([-1, 1, 1])*le_bottom + np.array(
                        [0, -5*skin_thickness, frame_length]), 8*skin_thickness, 5*skin_thickness))

            wire_points += center_points
        return meshes, wire_points

    def build_wires(self, wire_points):
        prev_t_f = None
        prev_b_f = None
        prev_t_r = None
        prev_b_r = None
        vertices = []
        edges = []

        for tf, bf, tr, br in zip(wire_points[0::4], wire_points[1::4], wire_points[2::4], wire_points[3::4]):
            curr_t_f = len(vertices)
            vertices.append(np.array(tf))
            curr_b_f = curr_t_f+1
            vertices.append(np.array(bf))
            if prev_t_f is not None:
                edges.append(np.array([prev_t_f, curr_b_f]))
            if prev_b_f is not None:
                edges.append(np.array([prev_b_f, curr_t_f]))
            prev_t_f = curr_t_f
            prev_b_f = curr_b_f

            curr_t_r = len(vertices)
            vertices.append(np.array(tr))
            curr_b_r = curr_t_r+1
            vertices.append(np.array(br))
            if prev_t_r is not None:
                edges.append(np.array([prev_t_r, curr_b_r]))
            if prev_b_r is not None:
                edges.append(np.array([prev_b_r, curr_t_r]))
            prev_t_r = curr_t_r
            prev_b_r = curr_b_r

        vertices = np.vstack(vertices)
        edges = np.vstack(edges)

        mesh1 = inflateAndMerge(vertices, edges, wires[2], 3)

        vertices[:, 0] *= -1
        mesh2 = inflateAndMerge(vertices, edges, wires[2], 3)

        return pymesh.merge_meshes([mesh1, mesh2])

    def build_vtruss(self, size):
        topw = self.wings[0] if len(
            self.wings) >= 1 else self.default_wing
        height = max(topw.height.value+1, WingHeight.PARASOL.value) * size
        frame_corner = size/math.sqrt(2)
        lt = 0
        lb = frame_corner
        le_top, le_bottom, chord_top, chord_bottom = self.strut_points(
            size, lt, lb, force_bottom=frame_corner)
        le_top = np.array([0, height, 0])
        wire_points = np.array([
            le_top+[0, 0, frame_length/2],
            le_bottom,
            le_bottom * [-1, 1, 1],
            le_bottom + [0, 0, chord_bottom],
            (le_bottom + [0, 0, chord_bottom]) * [-1, 1, 1],
        ])
        edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        frame = inflateAndMerge(np.array(wire_points),
                                edges, (self.num_trussv+1)*skin_thickness, 8)
        wire_points = [le_top+[0, 0, frame_length/2]]
        if topw.sweep > 0 and topw.sweep < 80:
            for s in range(1, 10):
                wire_points.append(topw.half_chord(topw.span/2*s/9))
                wire_points.append(topw.half_chord(
                    topw.span/2*s/9) * [-1, 1, 1])
        else:
            for s in range(1, 10):
                wire_points.append(topw.half_chord(math.pi/2*s/9))
                wire_points.append(topw.half_chord(math.pi/2*s/9) * [-1, 1, 1])
        edges = np.array([[0, i] for i in range(1, len(wire_points))])
        wires = inflateAndMerge(np.vstack(wire_points),
                                edges, 0.5*skin_thickness, 8)
        return [frame, wires]


class Strut():
    def __init__(self, le_top, le_bottom, chord_top, chord_bottom, gear=False, le_top2=[], le_bottom2=[]) -> None:
        self.le_top = np.array(le_top)
        self.le_bottom = np.array(le_bottom)
        self.chord_top = chord_top
        self.chord_bottom = chord_bottom
        self.gear = gear
        self.le_top2 = np.array(le_top2)
        self.le_bottom2 = np.array(le_bottom2)

    def build(self, s):
        if s == StrutType.PARALLEL:
            m, wp = self.parallel()
        elif s == StrutType.V:
            m, wp = self.v()
        elif s == StrutType.N:
            m, wp = self.n()
        elif s == StrutType.I:
            m, wp = self.i()
        elif s == StrutType.W:
            m, wp = self.w()
        elif s == StrutType.SINGLE:
            m, wp = self.single()
        elif s == StrutType.STAR:
            m, wp = self.star()
        elif s == StrutType.NONE:
            m = pymesh.Mesh()
            wp = []
        else:
            raise "Not Implemented Error."
        return m, wp

    def parallel(self):
        meshes = []
        wire_points = []

        pt = self.le_top+[0, 0, self.chord_top/5]
        pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        wire_points += [pt.copy(), pb.copy()]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pt += [0, 0, 3*self.chord_top/5]
        pb += [0, 0, 3*self.chord_bottom/5]
        wire_points += [pt.copy(), pb.copy()]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))

        pt = self.le_top+[0, 0, self.chord_top/5]
        pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        pt *= [-1, 1, 1]
        pb *= [-1, 1, 1]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pt += [0, 0, 3*self.chord_top/5]
        pb += [0, 0, 3*self.chord_bottom/5]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))

        return pymesh.merge_meshes(meshes), wire_points

    def v(self):
        if self.gear:
            thickness = 2*skin_thickness
        else:
            thickness = skin_thickness
        meshes = []
        wire_points = []

        pt = self.le_top+[0, 0, self.chord_top/5]
        if not self.gear:
            pb = self.le_bottom+[0, 0, self.chord_bottom/2]
        else:
            pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        wire_points += [pt.copy(), pb.copy()]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, thickness, thickness))
        pt += [0, 0, self.chord_top/3]
        wire_points += [pt.copy(), pb.copy()]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, thickness, thickness))

        pt = self.le_top+[0, 0, self.chord_top/5]
        if not self.gear:
            pb = self.le_bottom+[0, 0, self.chord_bottom/2]
        else:
            pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        pt *= [-1, 1, 1]
        pb *= [-1, 1, 1]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, thickness, thickness))
        pt += [0, 0, 3*self.chord_top/5]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, thickness, thickness))
        return pymesh.merge_meshes(meshes), wire_points

    def n(self):
        meshes = []
        wire_points = []

        pt = self.le_top+[0, 0, self.chord_top/5]
        pb = self.le_bottom+[0, 0, self. chord_bottom/5]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        wire_points += [pt.copy(), pb.copy()]
        pb += [0, 0, 3*self.chord_bottom/5]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pt += [0, 0, 3*self.chord_top/5]
        wire_points += [pt.copy(), pb.copy()]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))

        pt = self.le_top+[0, 0, self.chord_top/5]
        pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        pt *= [-1, 1, 1]
        pb *= [-1, 1, 1]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pb += [0, 0, 3*self.chord_bottom/5]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pt += [0, 0, 3*self.chord_top/5]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        return pymesh.merge_meshes(meshes), wire_points

    def i(self):
        meshes = []
        wire_points = []

        pt = self.le_top+[0, 0, self.chord_top/5]
        pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        wire_points += [pt.copy(), pb.copy()]

        af_pts = transform_pts(symm_airfoil, rotation=[0, 0, 90], scale=[
            10*skin_thickness, 10*skin_thickness, 10*skin_thickness])
        b_pts = transform_pts(af_pts, translation=pb-pt)

        vertices = np.vstack((af_pts, b_pts))
        tri = pymesh.tetgen()
        tri.points = vertices
        tri.verbosity = 0
        tri.run()
        meshes.append(transform(tri.mesh, translation=pt))
        wire_points += [pt.copy(), pb.copy()]

        pt = self.le_top+[0, 0, self.chord_top/5]
        pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        pt *= [-1, 1, 1]
        pb *= [-1, 1, 1]
        b_pts = transform_pts(af_pts, translation=pb-pt)

        vertices = np.vstack((af_pts, b_pts))
        tri = pymesh.tetgen()
        tri.points = vertices
        tri.verbosity = 0
        tri.run()
        meshes.append(transform(tri.mesh, translation=pt))
        return pymesh.merge_meshes(meshes), wire_points

    def w(self):
        meshes = []
        wire_points = []

        pt = self.le_top+[0, 0, self.chord_top/6]
        pb = self.le_bottom+[0, 0, 2*self.chord_bottom/6]
        wire_points += [pt.copy(), pb.copy()]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pt += [0, 0, 2*self.chord_top/6]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pb += [0, 0, 2*self.chord_bottom/6]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pt += [0, 0, 2*self.chord_top/6]
        wire_points += [pt.copy(), pb.copy()]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))

        pt = self.le_top+[0, 0, self.chord_top/6]
        pb = self.le_bottom+[0, 0, 2*self.chord_bottom/6]
        pt *= [-1, 1, 1]
        pb *= [-1, 1, 1]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pt += [0, 0, 2*self.chord_top/6]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pb += [0, 0, 2*self.chord_bottom/6]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        pt += [0, 0, 2*self.chord_top/6]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))

        return pymesh.merge_meshes(meshes), wire_points

    def single(self):
        meshes = []
        wire_points = []

        pt = self.le_top+[0, 0, self.chord_top/4]
        pb = self.le_bottom+[0, 0, self.chord_bottom/4]
        wire_points += [pt.copy(), pb.copy()]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, skin_thickness, skin_thickness))
        wire_points += [pt.copy(), pb.copy()]

        pt = self.le_top+[0, 0, self.chord_top/4]
        pb = self.le_bottom+[0, 0, self.chord_bottom/4]
        pt *= [-1, 1, 1]
        pb *= [-1, 1, 1]
        meshes.append(pymesh.generate_cylinder(
            pt, pb, 2*skin_thickness, 2*skin_thickness))
        return pymesh.merge_meshes(meshes), wire_points

    def star(self):
        meshes = []
        wire_points = []
        pt = self.le_top+[0, 0, self.chord_top/5]
        pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        wire_points += [pt.copy(), pb.copy()]
        pt += [0, 0, 3*self.chord_top/5]
        pb += [0, 0, 3*self.chord_bottom/5]
        wire_points += [pt.copy(), pb.copy()]

        pt = self.le_top2+[0, 0, self.chord_top/5]
        pb = self.le_bottom2+[0, 0, self.chord_bottom/5]
        wire_points += [pt.copy(), pb.copy()]
        pt += [0, 0, 3*self.chord_top/5]
        pb += [0, 0, 3*self.chord_bottom/5]
        wire_points += [pt.copy(), pb.copy()]

        edges = np.array([[0, 7], [1, 6], [2, 5], [3, 4]], np.int32)
        meshes.append(inflateAndMerge(
            np.array(wire_points), edges, 3*skin_thickness, 8))

        pt = self.le_top+[0, 0, self.chord_top/5]
        pb = self.le_bottom+[0, 0, self.chord_bottom/5]
        pt *= [-1, 1, 1]
        pb *= [-1, 1, 1]
        pt += [0, 0, 3*self.chord_top/5]
        pb += [0, 0, 3*self.chord_bottom/5]
        pt = self.le_top2+[0, 0, self.chord_top/5]
        pb = self.le_bottom2+[0, 0, self.chord_bottom/5]
        pt *= [-1, 1, 1]
        pb *= [-1, 1, 1]
        pt += [0, 0, 3*self.chord_top/5]
        pb += [0, 0, 3*self.chord_bottom/5]
        meshes.append(inflateAndMerge(
            [-1, 1, 1]*np.array(wire_points), edges, 3*skin_thickness, 8))

        return pymesh.merge_meshes(meshes), wire_points


class Tail(Part):
    def __init__(self, stab_type, nHStab, nVStab, sizeFactor=1, spanFactor=1, htipshape=TipShape.SFRONT, vtipshape=TipShape.CIRCLE) -> None:
        super().__init__()
        self.stab_type = stab_type
        self.nHStab = nHStab
        self.nVStab = nVStab
        self.sizeFactor = sizeFactor
        self.spanFactor = spanFactor
        self.area = 0
        self.span = 0
        self.htipshape = htipshape
        self.vtipshape = vtipshape

    def build(self, _, size):
        if self.stab_type == StabilizerType.NORMAL:
            tail_mesh = self.make_normal_tail(
                self.nHStab, self.nVStab, size)
        elif self.stab_type == StabilizerType.TTAIL:
            tail_mesh = self.make_single_ttail(self.nVStab)
        elif self.stab_type == StabilizerType.VTAIL:
            tail_mesh = self.make_single_vtail()
        else:
            raise "Not Implemented Yet"

        length = tail_mesh.bbox[1][2]
        return tail_mesh, length, None, None

    def set_wing_stats(self, area, span):
        self.area = area*self.sizeFactor
        self.span = span*self.spanFactor

    def make_single_HStab(self, wing_area, wing_span, nHStab):
        area = wing_area/8/nHStab
        span = wing_span/4

        hstab_meshes = []

        w = Wing((WingHeight.MID, area, span, 0, 0,
                 self.htipshape), symm_airfoil)
        w.set_size(1, 0)
        r, l = w.make_wing(0)
        hstab_meshes.append(l)
        hstab_meshes.append(r)
        return pymesh.merge_meshes(hstab_meshes), span, area/span

    def make_single_VStab(self, vstab_span, chord):
        w = Wing((WingHeight.MID, vstab_span *
                  chord, vstab_span, 0, 0, self.vtipshape), symm_airfoil)
        w.set_size(1, 0)
        r, l = w.make_wing(0)
        return transform(l, rotation=[0, 0, 90])

    def make_single_ttail(self, nVStab, size):
        tail_meshes = []
        hstab_mesh, hstab_span, hstab_chord = self.make_single_HStab(
            self.area/2, self.span, 1)
        vstab_span = hstab_span/2
        vstab_mesh = self.make_single_VStab(vstab_span, hstab_chord)

        if nVStab == 1:
            tail_meshes.append(vstab_mesh)
        else:
            for v in range(nVStab):
                sideways = -hstab_span/2 + v*hstab_span/(nVStab-1)
                tail_meshes.append(
                    transform(vstab_mesh, translation=[sideways, 0, 0]))

        tail_meshes.append(
            transform(hstab_mesh, translation=[0, 0.8*vstab_span, 0]))
        tail_meshes.append(pymesh.generate_cylinder(
            [0, 0, 0], [0, 0, hstab_chord], size, 0, 128))
        return pymesh.merge_meshes(tail_meshes)

    def make_single_vtail(self):
        area = self.area/5
        span = self.span/2

        vtail_meshes = []
        w = Wing((WingHeight.MID, area, span, 0, 0,
                 self.vtipshape), symm_airfoil)
        w.set_size(1, 0)
        r, l = w.make_wing(0)
        vtail_meshes.append(transform(l, rotation=[0, 0, 45]))
        vtail_meshes.append(transform(r, rotation=[0, 0, -45]))
        vtail_meshes.append(pymesh.generate_cylinder(
            [0, 0, 0], [0, 0, area/span], tail_width, 0, 128))
        return pymesh.merge_meshes(vtail_meshes)

    def make_normal_tail(self, nHStab, nVStab, size):
        tail_meshes = []

        if self.nHStab > 0:
            hstab_mesh, hstab_span, hstab_chord = self.make_single_HStab(
                self.area, self.span, self.nHStab)
            vstab_span = hstab_span
        else:
            vstab_span = self.area/4
            hstab_chord = self.area/(2*self.span)

        if nHStab > 1:
            vstab_span *= nHStab-1

        if self.nVStab > 0:
            vstab_mesh = self.make_single_VStab(vstab_span, hstab_chord)

        if nVStab == 1:
            tail_meshes.append(vstab_mesh)
        else:
            for v in range(nVStab):
                sideways = -hstab_span/2 + v*hstab_span/(nVStab-1)
                tail_meshes.append(
                    transform(vstab_mesh, translation=[sideways, 0, 0]))

        if nHStab == 1:
            tail_meshes.append(hstab_mesh)
        else:
            for h in range(nHStab):
                vert = h*vstab_span/(nHStab-1)
                tail_meshes.append(
                    transform(hstab_mesh, translation=[0, vert, 0]))
        tail_meshes.append(pymesh.generate_cylinder(
            [0, 0, 0], [0, 0, hstab_chord], tail_width+skin_thickness, 0, 128))
        return pymesh.merge_meshes(tail_meshes)


class Slice(object):
    def __init__(self, parts=[]) -> None:
        self.parts = []
        for type, args in parts:
            if isinstance(args, Part):
                part = args
            elif type == StackType.SHAPE:
                part = Shape(*args)
            elif type == StackType.PROPELLER:
                part = Propeller(*args)
            elif type == StackType.ENGINE:
                part = Engine(*args)
            elif type == StackType.SECTION:
                part = Section(*args)
            elif type == StackType.TAILSECTION:
                part = TailSection(*args)
            elif type == StackType.WINGBANK:
                part = WingBank(*args)
            elif type == StackType.TAIL:
                part = Tail(*args)
            else:
                print("{} not implemented".format(type))
            self.parts.extend(part if isinstance(part, list) else [part])

    def build(self, frame_stack_point, shape, size):
        meshes = []
        stack_increase = 0
        new_size = -1
        for idx, part in enumerate(self.parts):
            print("Part {}".format(idx))
            mesh, length, newshape, nsize = part.build(shape, size)
            if mesh is not None:
                meshes.append(transform(mesh, translation=[
                              0, 0, frame_stack_point]))
            if length is not None:
                stack_increase = max(stack_increase, length)
            if newshape is not None:
                shape = newshape
            if nsize is not None:
                new_size = max(new_size, nsize)
        if new_size < 0:
            new_size = size
        return meshes, stack_increase, shape, new_size

    def wing_stats(self):
        area = 0
        span = 0
        for p in self.parts:
            parea, pspan = p.wing_stats()
            area += parea
            span = max(span, pspan)
        return area, span

    def set_wing_stats(self, a, s):
        for part in self.parts:
            part.set_wing_stats(a, s)


class Stack(object):
    def __init__(self, slices=[]) -> None:
        self.slices = [Slice(s) for s in slices]

    def wing_stats(self):
        area = 0
        span = 0
        for s in self.slices:
            a, w = s.wing_stats()
            area += a
            span = max(span, w)
        return area, span

    def build(self):
        shape = FrameShape.CYLINDER
        stack_parts = []
        frame_stack_point = 0
        size = frame_width
        for idx, slice in enumerate(self.slices):
            print("Slice {}".format(idx))
            parts, length, shape, size = slice.build(
                frame_stack_point, shape, size)
            stack_parts += parts
            frame_stack_point += length
        return stack_parts

    def set_wing_stats(self, a, s):
        for slice in self.slices:
            slice.set_wing_stats(a, s)


class Plane(object):
    def __init__(self, stacks=[], offsets=[]) -> None:
        self.stacks = [Stack(s) for s in stacks]
        self.stack_offsets = offsets
        a, s = self.wing_stats()
        for stack in self.stacks:
            stack.set_wing_stats(a, s)

    def wing_stats(self):
        area = 0
        span = 0
        for s in self.stacks:
            a, w = s.wing_stats()
            area += a
            span = max(span, w)
        return area, span

    def Build(self):
        airplane_parts = []
        for idx, dat in enumerate(zip(self.stacks, self.stack_offsets)):
            stack, offset = dat
            print("Stack {}".format(idx))
            stack_parts = stack.build()
            stack_mesh = pymesh.merge_meshes(stack_parts)
            airplane_parts.append(transform(stack_mesh, translation=offset))
        return pymesh.merge_meshes(airplane_parts)


def make_firewall(size, shape):
    vertices = np.vstack(
        (frame_types[shape]*size, frame_types[shape]*size+np.array([0, 0, skin_thickness])))
    tri = pymesh.tetgen()
    tri.points = vertices
    tri.verbosity = 0
    tri.run()
    return tri.mesh  # output triangulation.


def make_section(length, front_size, rear_size, front_shape, rear_shape, skin=Skin.CLOTH, firewall_front=False, firewall_rear=False, opentop=False):
    pieces = []
    if skin == Skin.CLOTH:
        pieces.append(make_both(front_shape, rear_shape,
                      front_size, rear_size, length, opentop))
    else:
        pieces.append(make_frame(front_shape, rear_shape,
                      front_size, rear_size, length, opentop))

    if firewall_front:
        pieces.append(make_firewall(front_size, front_shape))

    if firewall_rear:
        pieces.append(transform(make_firewall(
            rear_size, rear_shape), translation=[0, 0, length]))

    section = pymesh.merge_meshes(pieces)
    return section


def Save(name, plane):
    print('Saving {}'.format(name))
    with open('/script/{}.obj'.format(name), 'w') as f:
        f.write("# OBJ file\n")
        fmt = 'v ' + ' '.join(['{}']*plane.vertices.shape[1])
        fmt = '\n'.join([fmt]*plane.vertices.shape[0])
        f.write(fmt.format(*tuple(plane.vertices.ravel())))
        faces = plane.faces + 1
        f.write('\n')
        fmt = 'f ' + ' '.join(['{}']*faces.shape[1])
        fmt = '\n'.join([fmt]*faces.shape[0])
        f.write(fmt.format(*tuple(faces.ravel())))

    print('Finished Saving {}'.format(name))


name = 'Blobluft'
wings = [[Wing((WingHeight.PARASOL, 8, 8, 0, 5, TipShape.SFRONT), airfoil),
          Wing((WingHeight.LOW, 8, 4, 0, 5, TipShape.ROUNDED), airfoil)]]
engines = []
fuselage = [Section(FrameShape.CYLINDER, Skin.NAKED, Cockpit.WINDSCREEN, True),
            Section(FrameShape.CYLINDER, Skin.NAKED, Cockpit.NONE)]
stacks = [
    [
        [(StackType.SHAPE, Shape(FrameShape.CYLINDER, size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.TWOBLADE))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.FORWARD))],
        [(StackType.SECTION, fuselage[0]),
         (StackType.WINGBANK, WingBank(wings[0], Stagger.POSITIVE, cabane=StrutType.PARALLEL, struts=[
          StrutType.PARALLEL, StrutType.PARALLEL], wires=True)),
         (StackType.WINGBANK, WingBank([], Stagger.UNSTAGGERED, StrutType.NONE, [], GearType.WHEEL))],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.TAILSECTION, TailSection(2, Skin.NAKED, rear_size=tail_width))],
        [(StackType.TAIL, Tail(StabilizerType.NORMAL, 1, 1))],
    ],
]
offsets = [[0, 0, 0]]

aircraft = Plane(stacks=stacks, offsets=offsets)
plane = aircraft.Build()
Save(name, plane)


name = 'Schnappchenchuss'
wings = [[Wing((WingHeight.PARASOL, 8, 8, 0, 0, TipShape.SQUARE), airfoil),
          Wing((WingHeight.MID, 4, 4, 0, 0, TipShape.SQUARE), airfoil),
         Wing((WingHeight.GEAR, 10, 8, 0, 0, TipShape.SQUARE), symm_airfoil)]]
fuselage = [Section(FrameShape.CYLINDER, Skin.CLOTH, Cockpit.OPEN, opentop=True),
            Section(FrameShape.CYLINDER, Skin.CLOTH, Cockpit.NONE)]
stacks = [
    [
        [(StackType.SHAPE, Shape(FrameShape.CYLINDER, size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.TWOBLADE))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.FORWARD, rear_firewall=True))],
        [(StackType.SECTION, fuselage[0]),
         (StackType.WINGBANK, WingBank(wings[0], Stagger.NEGATIVE, cabane=StrutType.PARALLEL, struts=[
          StrutType.PARALLEL, StrutType.N], gear=GearType.WHEEL, wires=True))],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.TAILSECTION, TailSection(2, Skin.CLOTH, rear_size=tail_width))],
        [(StackType.TAIL, Tail(StabilizerType.NORMAL, 1, 1,
          htipshape=TipShape.SFRONT, vtipshape=TipShape.SFRONT))],
    ],
]
offsets = [[0, 0, 0]]

aircraft = Plane(stacks=stacks, offsets=offsets)
plane = aircraft.Build()
Save(name, plane)


name = 'Bunderfalke'
wings = [[Wing((WingHeight.MID, 8*frame_length, 8, 0, 2500, TipShape.SQUARE), (airfoil, ellipse_airfoil))],
         [Wing((WingHeight.MID, 8*frame_length, 8, 0, -2500, TipShape.SQUARE), (airfoil, ellipse_airfoil))]]
fuselage = [Section(FrameShape.MIXED, Skin.CLOTH, Cockpit.OPEN, opentop=True),
            Section(FrameShape.MIXED, Skin.CLOTH, Cockpit.NONE, opentop=False),
            Section(FrameShape.MIXED, Skin.NAKED, Cockpit.NONE, opentop=False)]
stacks = [
    [
        [(StackType.SHAPE, Shape(FrameShape.MIXED, size=frame_width))],
        [(StackType.SECTION, fuselage[1]),
         (StackType.WINGBANK, WingBank(wings[0], Stagger.UNSTAGGERED, cabane=None, struts=[StrutType.TRUSSV
                                                                                           ], gear=GearType.WHEEL, wires=False))
         ],
        [(StackType.SECTION, fuselage[0])],
        [(StackType.TAILSECTION, TailSection(
            2, Skin.CLOTH, rear_shape=FrameShape.MIXED))],
        [(StackType.SECTION, fuselage[1]),
            (StackType.WINGBANK, WingBank(wings[1], Stagger.UNSTAGGERED, cabane=None, struts=[StrutType.TRUSSV], gear=GearType.WHEEL, wires=False))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
                                   EngineMounting.EXPOSED, EngineDirection.BACKWARD))],
        [(StackType.PROPELLER, Propeller(PropellerType.THREEBLADE))],
    ],
    [
        [(StackType.TAIL, Tail(StabilizerType.NORMAL, 0, 1, sizeFactor=0.5,
                               vtipshape=TipShape.ROUNDED)), ]
    ]
]
offsets = [[0, 0, 0], [0, 0, 4.5*frame_length]]

aircraft = Plane(stacks=stacks, offsets=offsets)
plane = aircraft.Build()
Save(name, plane)


name = 'Brechenstein'
wings = [[Wing((WingHeight.SHOULDER, 12, 12, 0, 5, TipShape.SQUARE), airfoil),
          Wing((WingHeight.LOW, 12, 12, 0, 5, TipShape.SQUARE), airfoil)]]
engines = []
fuselage = [Section(FrameShape.BOX, Skin.CLOTH, Cockpit.OPEN, True),
            Section(FrameShape.BOX, Skin.CLOTH, Cockpit.NONE)]
stacks = [
    [
        [(StackType.SHAPE, Shape(FrameShape.CYLINDER, size=0.8*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.THREEBLADE))],
        [(StackType.ENGINE, Engine(EngineType.ROTARY,
          EngineMounting.EXPOSED, EngineDirection.FORWARD))],
        [(StackType.SECTION, fuselage[0]),
         (StackType.WINGBANK, WingBank(wings[0], Stagger.XPOSITIVE, cabane=StrutType.PARALLEL, struts=[
          StrutType.PARALLEL], wires=True, gear=GearType.WHEEL)), ],
        [(StackType.SECTION, fuselage[0]), ],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.TAILSECTION, TailSection(3, Skin.CLOTH, rear_size=tail_width))],
        [(StackType.TAIL, Tail(StabilizerType.NORMAL, 1, 1,
          htipshape=TipShape.SFRONT, vtipshape=TipShape.CIRCLE))],
    ],
]
offsets = [[0, 0, 0]]

aircraft = Plane(stacks=stacks, offsets=offsets)
plane = aircraft.Build()
Save(name, plane)


name = 'Skyborn Special'
wings = [[Wing((WingHeight.MID, 27, 9, 0, 5, TipShape.SFRONT), airfoil)]]
engines = []
fuselage = [Section(FrameShape.CYLINDER, Skin.CLOTH, Cockpit.OPEN, True),
            Section(FrameShape.BOX, Skin.CLOTH, Cockpit.NONE)]
stacks = [
    [
        [(StackType.SHAPE, Shape(FrameShape.CYLINDER, size=frame_width))],
        [(StackType.SECTION, fuselage[0]),
         (StackType.WINGBANK, WingBank(wings[0], Stagger.UNSTAGGERED, cabane=None, struts=[StrutType.TRUSSV, StrutType.TRUSSV], wires=False, gear=GearType.SKID)), ],
        [(StackType.SECTION, fuselage[1]), ],
        [(StackType.TAILSECTION, TailSection(2, Skin.CLOTH, rear_size=tail_width))],
        [(StackType.TAIL, Tail(StabilizerType.NORMAL, 1, 1,
          htipshape=TipShape.SFRONT, vtipshape=TipShape.SFRONT))],
    ],
    [
        [(StackType.SHAPE, Shape(FrameShape.CYLINDER, size=0.5*frame_width))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.BACKWARD, new_size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.FOURBLADE))],
    ],
    [
        [(StackType.SHAPE, Shape(FrameShape.CYLINDER, size=0.5*frame_width))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.BACKWARD, new_size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.FOURBLADE))],
    ],
]
offsets = [[0, 0, 0], [-2, 0, 1.6*frame_length], [2, 0, 1.6*frame_length]]

aircraft = Plane(stacks=stacks, offsets=offsets)
plane = aircraft.Build()
Save(name, plane)


name = 'Super SeePfau'
wings = [[Wing((WingHeight.SHOULDER, 10, 11, 0, 0, TipShape.SQUARE), airfoil),
          Wing((WingHeight.LOW, 10, 11, 0, 0, TipShape.SQUARE), airfoil), ]]
fuselage = [Section(FrameShape.CYLINDER, Skin.CLOTH, Cockpit.OPEN, opentop=True),
            Section(FrameShape.CYLINDER, Skin.CLOTH, Cockpit.NONE)]
stacks = [
    [
        [(StackType.SHAPE, Shape(FrameShape.CYLINDER, size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.TWOBLADE))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.FORWARD, rear_firewall=True))],
        [(StackType.SECTION, fuselage[0]),
         (StackType.WINGBANK, WingBank(wings[0], Stagger.XPOSITIVE, cabane=StrutType.PARALLEL, struts=[
          StrutType.PARALLEL], gear=GearType.FLOATS, wires=True))],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.TAILSECTION, TailSection(1, Skin.CLOTH, rear_size=tail_width))],
        [(StackType.TAIL, Tail(StabilizerType.NORMAL, 1, 1,
          htipshape=TipShape.ROUNDED, vtipshape=TipShape.CIRCLE))],
    ],
]
offsets = [[0, 0, 0]]

aircraft = Plane(stacks=stacks, offsets=offsets)
plane = aircraft.Build()
Save(name, plane)


name = 'Barons Bomber'
wings = [[Wing((WingHeight.PARASOL, 30, 21, 0, 5, TipShape.SQUARE), airfoil),
          Wing((WingHeight.LOW, 30, 21, 0, 5, TipShape.SQUARE), airfoil), ]]
fuselage = [Section(FrameShape.BOX, Skin.CLOTH, Cockpit.OPEN, opentop=True),
            Section(FrameShape.BOX, Skin.CLOTH, Cockpit.NONE)]
stacks = [
    [
        [(StackType.SHAPE, Shape(FrameShape.BOX, size=frame_width))],
        [(StackType.SECTION, fuselage[0]), ],
        [(StackType.SECTION, fuselage[0]),
         (StackType.WINGBANK, WingBank(wings[0], Stagger.XPOSITIVE, cabane=StrutType.STAR, struts=[
          StrutType.PARALLEL, StrutType.PARALLEL, StrutType.PARALLEL, StrutType.SINGLE, StrutType.TRUSSV, StrutType.TRUSSV], gear=GearType.WHEEL, wires=True))],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.TAILSECTION, TailSection(3, Skin.CLOTH, rear_size=tail_width))],
        [(StackType.TAIL, Tail(StabilizerType.NORMAL, 1, 1, sizeFactor=0.75, spanFactor=0.75,
          htipshape=TipShape.SFRONT, vtipshape=TipShape.ROUNDED))],
    ],
    [
        [(StackType.SHAPE, Shape(FrameShape.BOX, size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.FOURBLADE))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.FORWARD, new_size=0.5*frame_width))],
    ],
    [
        [(StackType.SHAPE, Shape(FrameShape.BOX, size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.FOURBLADE))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.FORWARD, new_size=0.5*frame_width))],
    ],
]
offsets = [[0, 0, 0], [-5, 0, 1.25*frame_length], [5, 0, 1.25*frame_length]]

aircraft = Plane(stacks=stacks, offsets=offsets)
plane = aircraft.Build()
Save(name, plane)


name = 'Bergziegel'
wings = [[Wing((WingHeight.SHOULDER, 25, 10, 0, 5, TipShape.SQUARE), airfoil),
          Wing((WingHeight.LOW, 25, 10, 0, 5, TipShape.SQUARE), airfoil), ]]
fuselage = [Section(FrameShape.MIXED, Skin.CLOTH, Cockpit.OPEN, opentop=True, newsize=2*frame_width),
            Section(FrameShape.MIXED, Skin.CLOTH, Cockpit.NONE),
            Section(FrameShape.MIXED, Skin.CLOTH, Cockpit.NONE, newsize=frame_width)]
stacks = [
    [
        [(StackType.SHAPE, Shape(FrameShape.MIXED, size=frame_width))],
        [(StackType.SECTION, fuselage[0]), ],
        [(StackType.SECTION, fuselage[1]),
         (StackType.WINGBANK, WingBank(wings[0], Stagger.POSITIVE, cabane=StrutType.PARALLEL, struts=[
          StrutType.PARALLEL, StrutType.PARALLEL], gear=GearType.WHEEL, wires=True))],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.SECTION, fuselage[1])],
        [(StackType.SECTION, fuselage[2])],
        [(StackType.TAILSECTION, TailSection(3, Skin.CLOTH, rear_size=tail_width))],
        [(StackType.TAIL, Tail(StabilizerType.NORMAL, 1, 1, sizeFactor=1, spanFactor=1,
          htipshape=TipShape.SQUARE, vtipshape=TipShape.SQUARE))],
    ],
    [
        [(StackType.SHAPE, Shape(FrameShape.BOX, size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.FOURBLADE))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.FORWARD, new_size=0.5*frame_width))],
    ],
    [
        [(StackType.SHAPE, Shape(FrameShape.BOX, size=0.5*frame_width))],
        [(StackType.PROPELLER, Propeller(PropellerType.FOURBLADE))],
        [(StackType.ENGINE, Engine(EngineType.INLINE,
          EngineMounting.EXPOSED, EngineDirection.FORWARD, new_size=0.5*frame_width))],
    ],
]
offsets = [[0, 0, 0], [-2.75, -0.2, frame_length], [2.75, -0.2, frame_length]]

aircraft = Plane(stacks=stacks, offsets=offsets)
plane = aircraft.Build()
Save(name, plane)

# Nose Caps
# Wing Seam?
# Wing TrussesH
# Landing Skid
# Float tips
