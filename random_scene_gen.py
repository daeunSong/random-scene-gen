from random import *
import sys
import pickle

import matplotlib.pyplot as plt    
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import numpy as np


## tile types: normal, vertical-long, horizontal-long, empty
TILE_TYPES = ['n', 'n', 'n', 'n', 'v', 'v', 'h', 'h', 'e']

RAND_W = False
RAND_H = False 
RAND_HEIGHT = False 
RAND_PITCH = False
RAND_YAW = False

MAX_W = 0.3
MAX_H = 0.4
MAX_GAP = 0.05
HEIGHT_SAMPLE = [0., 0.5, 1.0]

### set up a random grid with the type of the tiles for rubbles
def set_grid ():
    num_row = randint(3,7) #4
    num_col = randint(3,7) #4
    grid = [[None for j in range(num_col)] for i in range(num_row)]
    grid_indices = [[None for j in range(num_col)] for i in range(num_row)]
  
    for i in range(num_row):
        for j in range(num_col):
            if grid[i][j] is None:
                el = sample(TILE_TYPES, 1)[0]
                if el is 'v':
                    if j == num_col-1 or grid[i][j+1] == 'h':  
                        el = 'n'
                    else: 
                        grid[i][j+1] = 'v'
                elif el is 'h':
                    if i == num_row-1 or grid[i+1][j] == 'v':
                        el = 'n'
                    else:
                        grid[i+1][j] = 'h'
          
                grid[i][j] = el
    return grid

### scene gernation accordig to the generated grid
def scene_gen ():
    grid = set_grid() # grid for rubbles
    scene = [[]]
    
    p = np.zeros(3)
    w = round(uniform(0.2, MAX_W),2) if RAND_W else 0.15
    h = round(uniform(0.15, MAX_H),2) if RAND_H else 0.15
    gap = 0.05
    
    w_plane = (w+gap/2)*(len(grid)+1)-gap/2
    h_plane = (h+gap/2)*(len(grid[0])+1)-gap/2
    
    # w_plane = (w+gap/2)*len(grid)-gap/2
    # h_plane = (h+gap/2)*len(grid[0])-gap/2
    p[0] -= h_plane
    
    
    # generate plane
    plane, p = plane_gen(p, w_plane, h_plane, gap)
    scene = plane
    
    # generate bridge for 0.5 chance
    if uniform(0,1) < 0.5:
        bridge, p = bridge_gen(p, w_plane, h_plane, gap)
        scene += bridge
    # generate rubbles for 0.5 chance
    if uniform(0,1) < 0.5:
        rubbles, p = rubbles_gen(grid, p, w, h, gap)
        plane, p = plane_gen(p, w_plane, h_plane, gap)
        scene += rubbles + plane
    # generate stairs for 0.5 chance
    if uniform(0,1) < 0.5:
        stairs, p = stairs_gen (p, 0.4, 0.15, gap)
        plane, p = plane_gen(p, w_plane, h_plane, gap)
        scene += stairs + plane
        # generate bridge for 0.3 chance
        if uniform(0,1) < 0.3:
            p[0] -= h_plane*2 #+ 1.0
            p[1] -= w_plane*2
            bridge, p = bridge_inv_gen(p, w_plane, h_plane, gap)
            scene += bridge
        
    return scene

# rubbles gen according to the grid
def rubbles_gen (grid, p, w, h, gap):
    # p = zeros(3)
    p[0] += gap + h
    p[1] += (w + gap/2) * len(grid) - gap/2 -w
    p_origin = p
    surfaces = []
    # surfaces += [rectangle_gen([-h*2-gap, w*1.5+gap,0], w*3, h, height)]
  
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            height = sample(HEIGHT_SAMPLE, 1) if RAND_HEIGHT else 0.
            p = (np.array(p_origin) + np.array([2*h*j + gap*j, -2*w*i - gap*i, height])).tolist()
            if grid[i][j] is 'n': 
                surfaces += [random_rectangle_gen(p, w, h)]
            elif grid[i][j] is 'v': 
                p[0] += h + gap/2
                surfaces += [rectangle_gen(p, w, 2*h + gap/2)]
                grid[i][j+1] = 'e'
            elif grid[i][j] is 'h':
                p[1] -= (w + gap/2)
                surfaces += [rectangle_gen(p, 2*w + gap/2, h)]
                grid[i+1][j] = 'e'
    
    p = (np.array(p_origin) + np.array([len(grid[0])*(2*h+gap)-gap/2-h, len(grid)*(-w-gap/2)+w+gap/2, p_origin[2]])).tolist()
        
    return surfaces, p
    
def stairs_gen (p, w, h, gap):    
    num_stairs = randint(5,8)
    height_diff = 0.1
    surfaces = []
    p[0] += gap + h
    
    for i in range(num_stairs):
        surfaces += [rectangle_gen (p, w, h)]
        p[0] += 2*h; p[2] += height_diff 
    
    p[0] -= h; p[2] -= height_diff
    
    return surfaces, p

def plane_gen (p, w, h, gap):
    p[0] += h 
    surfaces = [rectangle_gen(p, w, h)]
    p[0] += h
    return surfaces, p

def bridge_gen (p, w, h, gap):
    p[0] += 1.5
    bridge = [rectangle_gen(p, 0.3, 1.5)]
    p[0] += 1.5
    plane2, p = plane_gen(p, w, h, gap)
    return bridge+plane2, p
    
def bridge_inv_gen (p, w, h, gap):
    plane1, p = plane_gen(p, w, h, gap)
    p[0] -= 1.5 + h*2
    bridge = [rectangle_gen(p, 0.3, 1.5)]
    p[0] -= 1.5 + h*2
    plane2, p = plane_gen(p, w, h, gap)
    return plane1+bridge+plane2, p
    

### normal rectangle generation 
def rectangle_gen (p, w, h):  
    p1 = (np.array(p) + np.array([-h, w, 0.])).tolist()
    p2 = (np.array(p) + np.array([h, w, 0.])).tolist()
    p3 = (np.array(p) + np.array([h, -w, 0.])).tolist()
    p4 = (np.array(p) + np.array([-h, -w, 0.])).tolist()
    return [p1, p2, p3, p4]

### rectangle generation with some transformations
def random_rectangle_gen (p, w, h):  
    pts = rectangle_gen(p, w, h)
  
    if RAND_PITCH:
        val = round(uniform(-0.05,0.05),2)
        if (randint(0,1) == 0):
              pts[2][2] += val; pts[3][2] += val
        else :
            pts[0][2] += val; pts[1][2] += val
    
    if RAND_YAW:
        val = round(uniform(-0.1,0.1),2)
        if (randint(0,1) == 0):
              pts[1][2] += val; pts[2][2] += val
        else :
            pts[0][2] += val; pts[3][2] += val
  
    return pts       

### TOOLS
def get_center_point (points):
    return np.array(points).mean(axis=0).tolist()
  
def get_dist (p1, p2):
    dist = 0
    for i in range (0, 3):
        dist += (p1[i] - p2[i])**2
    return sqrt(dist)

def cutList2D(l):
    return [el[:2] for el in l]

def list_to_arr (all_surfaces):
    all_surfaces_arr = []
    for surface in all_surfaces:
        all_surfaces_arr.append(np.array(surface).T)
    return all_surfaces_arr

def sort_by_dist (all_surfaces, index_list, p1):
    sorted_list = []
    for index in index_list:
        p2 = get_center_point(all_surfaces[index])
        dist = get_dist(p1, p2)
        sorted_list.append((index, dist))
    sorted_list.sort(key=lambda element:element[1])
    return [el[0] for el in sorted_list]


### problem generation with start/goal tiles
def problem_gen (surfaces):
    index_start = 0
    index_goal = len(surfaces)-1
    
    p_start = get_center_point(surfaces[index_start])
    p_goal = get_center_point(surfaces[index_goal])
    
    mins = [100000,100000,100000]
    maxs = [-100000,-100000,-100000]
    for surface in surfaces:
        surface = np.array(surface).T
        for i, points in enumerate(surface):
            mins[i] = min(points+[mins[i]])
            maxs[i] = max(points+[maxs[i]])
    
    return p_start, p_goal, mins, maxs

def save_pb (fileName, surfaces):
    p_start, p_goal, mins, maxs = problem_gen(surfaces)
    f = open(fileName,'w')
    f.write("%f %f %f\n" %(p_start[0], p_start[1], p_start[2]))
    f.write("%f %f %f\n" %(p_goal[0], p_goal[1], p_goal[2]))
    f.write("%f %f\n" %(mins[0], maxs[0]))
    f.write("%f %f\n" %(mins[1], maxs[1]))
    f.write("%f %f\n" %(mins[2], maxs[2]))
    f.close()
    

### plot generated surfaces
def plot_surface (points, ax, color_id = -1):
    xs = np.append(points[0,:] ,points[0,0] ).tolist()
    ys = np.append(points[1,:] ,points[1,0] ).tolist()
    zs = (np.append(points[2,:] ,points[2,0] ) - np.ones(len(xs))*0.005*color_id).tolist()
    colors = ['r','g','b','m','y','c']
    if color_id == -1: ax.plot(xs,ys,zs)
    else: ax.plot(xs,ys,zs,colors[color_id])
        
def draw_scene(surfaces, ax = None):
    colors = ['r','g','b','m','y','c']
    color_id = 0
    if ax is None:        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for surface in surfaces:
        plot_surface(np.array(surface).T, ax)
    plt.ion()
    return ax    

### save surfaces into mesh file
def save_obj (fileName, surfaces):
    i = 1
    f = open(fileName,'w')
    for surface in surfaces:
        for point in surface:
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n' .format(point[0], point[1], point[2]))
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n' .format(point[0], point[1], point[2]-0.1))
        f.write('f {0} {1} {2} {3} \n'.format(i, i+1, i+3, i+2))
        f.write('f {0} {1} {2} {3} \n'.format(i, i+1, i+7, i+6))
        f.write('f {0} {1} {2} {3} \n'.format(i, i+6, i+4, i+2))
        f.write('f {0} {1} {2} {3} \n'.format(i+1, i+7, i+5, i+3))
        f.write('f {0} {1} {2} {3} \n'.format(i+6, i+7, i+5, i+4))
        f.write('f {0} {1} {2} {3} \n'.format(i+4, i+5, i+3, i+2))
        i += 8
    f.close()

############# main ###################    

if __name__ == '__main__':
    all_surfaces = scene_gen()
    ax = draw_scene(all_surfaces)
    plt.show()
    fileName = sys.argv[1]
    save_obj("obj/"+fileName+".obj", all_surfaces)
    save_pb("pb/"+fileName, all_surfaces)

