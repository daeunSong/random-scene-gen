from random import *
import sys
import pickle

import matplotlib.pyplot as plt    
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import numpy as np

from tile_classes import BOTTOM, TOP, RIGHT, LEFT, VEC_POSITIONING, NAME_POSITIONING
from tile_classes import Entry, Tile, Tile_to_set, Link


# ========================================================================================
# ========================== GENENERATION OF BASIC ELEMENTS ==============================
# ========================================================================================

def plane_gen (p, w, h, gap):
    p[0] += h 
    surfaces = [rectangle_gen(p, w, h)]
    p[0] += h
    return surfaces, p

### normal rectangle generation 
def rectangle_gen (p, w, h):  
    p1 = (np.array(p) + np.array([-h, w, 0.])).tolist()
    p2 = (np.array(p) + np.array([h, w, 0.])).tolist()
    p3 = (np.array(p) + np.array([h, -w, 0.])).tolist()
    p4 = (np.array(p) + np.array([-h, -w, 0.])).tolist()
    return [p1, p2, p3, p4]   

### rectangle generation with some transformations V2
def random_rectangle_gen(p, w, h, dw=0.05, dh=0.05):
    pts = rectangle_gen(p, w, h)
    # PITCH
    val = round(uniform(-dw,dw),2)
    if (randint(0,1) == 0):
        pts[2][2] += val; pts[3][2] += val
    else :
        pts[0][2] += val; pts[1][2] += val
    # YAW
    val = round(uniform(-dh,dh),2)
    if (randint(0,1) == 0):
        pts[1][2] += val; pts[2][2] += val
    else :
        pts[0][2] += val; pts[3][2] += val
    return pts


### cube generation
def cube_gen(p, size_x, size_y, size_z):
    pts = np.array([[-1, -1, -1],
                    [1, -1, -1 ],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1 ],
                    [1, 1, 1],
                    [-1, 1, 1]])
    pts = (pts * np.array([size_x,size_y,size_z])).tolist()
    pts += np.array(p)
    surfaces = [  
                [pts[0],pts[1],pts[2],pts[3]],
                [pts[4],pts[5],pts[6],pts[7]],
                [pts[0],pts[1],pts[5],pts[4]],
                [pts[2],pts[3],pts[7],pts[6]],
                [pts[1],pts[2],pts[6],pts[5]],
                [pts[4],pts[7],pts[3],pts[0]]
               ]
    return surfaces

############## 

# ========================================================================================
# ============================== CREATE BASIC TILES ======================================
# ========================================================================================

# All tiles are squares.
# Tile size is [x,y], but in function above, h is X, w is Y.

# Types of tile : 'f'=flat / 'r'=rubbles / 's'=stairs / 'b'=bridge / 'o'=obstacle

# After each tile of type other than 'f', the entries open will have a tile 'f' or 'o' connected to it





# Tile generation functions

def tile_flat_gen(p, indices_tile, tile_size=[2.0,2.0]):
    surfaces = [rectangle_gen(p, h=tile_size[0]/2., w=tile_size[1]/2.)]
    # Tile entries are all at the same height and open
    return Tile(p, indices_tile, tile_size, surfaces)

def tile_rubbles_gen(p, indices_tile, tile_size=[2.0,2.0], bound_h=[0.25,0.35], bound_w=[0.25,0.35]):
    surfaces = []
    list_h, list_w = [], []
    # Bounds max of X and Y for this tile
    limit_tile_x, limit_tile_y = p[0]+tile_size[0]/2., p[1]+tile_size[1]/2.
    # Pos start correspond to bounds min of X and Y for this tile, plus the demi-size of first step
    p_start = [ p[0]-tile_size[0]/2., p[1]-tile_size[1]/2., p[2] ]
    dh, dw = 0.05, 0.05
    x = p_start[0]
    while (x<limit_tile_x):
        # Update X
        val_h = round(uniform(bound_h[0],bound_h[1]),2)
        dh_actual = dh
        if x+val_h>limit_tile_x:
            val_h = limit_tile_x-x
            dh_actual = 0.
        x += val_h/2.
        y = p_start[1]
        while (y<limit_tile_y):
            # Update Y
            val_w = round(uniform(bound_w[0],bound_w[1]),2)
            dw_actual = dw
            if y+val_w>limit_tile_y:
                val_w = limit_tile_y-y
                dw_actual = 0.
            y += val_w/2.
            # Create one rubble
            surfaces += [ random_rectangle_gen([x,y,p[2]], h=val_h/2., w=val_w/2., dh=dh_actual, dw=dw_actual) ]
            # Update Y again
            y += val_w/2.
        # Update X again
        x += val_h/2.
    # Tile entries are all at the same height and open
    return Tile(p, indices_tile, tile_size, surfaces, type='r')

def tile_bridge_gen(p, indices_tile, tile_size=[2.0,2.0], left_to_right=True, width_ramp=0.30, height_LR_or_BT=[None,None]):
    # I do not use bridge_gen, I recreate this function
    # This should always be connected to a flat tile
    if left_to_right:
        h = tile_size[0]
        w = width_ramp
    else:
        h = width_ramp
        w = tile_size[1]
    surfaces = [ rectangle_gen(p, h=h/2., w=w/2.) ]
    # If we force the height of left and right, the bridge will be in slope
    if (height_LR_or_BT[0] is not None) and (height_LR_or_BT[1] is not None):
        if left_to_right:
            print("LR BT : ",height_LR_or_BT)
            surfaces[0][0][2] = height_LR_or_BT[1] # Left
            surfaces[0][3][2] = height_LR_or_BT[1]
            surfaces[0][1][2] = height_LR_or_BT[0] # Right
            surfaces[0][2][2] = height_LR_or_BT[0]
        else:
            surfaces[0][0][2] = height_LR_or_BT[0] # Bottom
            surfaces[0][1][2] = height_LR_or_BT[0]
            surfaces[0][2][2] = height_LR_or_BT[1] # Top
            surfaces[0][3][2] = height_LR_or_BT[1]
    # Create tile
    tile = Tile(p, indices_tile, tile_size, surfaces, type='b')
    if left_to_right:
        tile.update_entry(positioning=LEFT  , height=surfaces[0][0][2], open=True)
        tile.update_entry(positioning=RIGHT , height=surfaces[0][1][2], open=True)
        tile.update_entry(positioning=BOTTOM, height=p[2], open=False)
        tile.update_entry(positioning=TOP   , height=p[2], open=False)
    else:
        tile.update_entry(positioning=BOTTOM, height=surfaces[0][0][2], open=True)
        tile.update_entry(positioning=TOP   , height=surfaces[0][2][2], open=True)
        tile.update_entry(positioning=LEFT  , height=p[2], open=False)
        tile.update_entry(positioning=RIGHT , height=p[2], open=False)
    return tile

# Height to climb can be negative if we want stairs to go down.
def tile_stairs_gen(p, indices_tile, tile_size=[2.0,2.0], positioniong_start=LEFT, height_start=0., height_arrival=1., width_step=[0.3,0.45], length_step=[0.6,1.0]):
    surfaces = []
    p_start = [ p[0]-tile_size[0]/2., p[1]-tile_size[1]/2., p[2] ]
    # Pick axis to orient stairs
    axis = 0
    if positioniong_start==BOTTOM or positioniong_start==TOP:
        axis = 1
    # Are the stairs going down or up
    # For now it's going to increase from p[2] to p[2]+height_to_climb on [p_left, p_right] or [p_bottom, p_top]
    # So we invert it if we start on right or top
    limit_tile_axis = p[axis]+tile_size[axis]/2.
    pos_actual = p[::]
    height_to_climb = height_arrival - height_start
    # Adjust starting height and value of climbing to orientation
    if positioniong_start==LEFT or positioniong_start==BOTTOM:
        pos_actual[2] = height_start
    else:
        pos_actual[2] = height_arrival
        height_to_climb = -height_to_climb
    # Set 2D position of stairs
    pos_axis = p_start[axis]
    width_next = round(uniform(width_step[0],width_step[1]),2)
    while pos_axis<limit_tile_axis:
        width_actual = width_next
        length_actual = round(uniform(length_step[0],length_step[1]),2)
        # Width of next step
        width_next = round(uniform(width_step[0],width_step[1]),2)
        #If the next step is too small, set actual step to the end
        if pos_axis+width_next/2.0+width_step[0]>limit_tile_axis:
            width_actual = limit_tile_axis-pos_axis
        # Update pos along axis
        pos_axis += width_actual/2.0
        # Create one step
        pos_actual[axis] = pos_axis
        if axis==0:
            surfaces += [ rectangle_gen(pos_actual, h=width_actual/2., w=length_actual/2.) ]
        else:
            surfaces += [ rectangle_gen(pos_actual, h=length_actual/2., w=width_actual/2.) ]
        # Update pos along axis again
        pos_axis += width_actual/2.0
    # Set height value of stairs
    for i in range(1, len(surfaces)):
        step_height = height_to_climb/len(surfaces)
        # Loop for points of surface
        for j in range(0,len(surfaces[i])):
            surfaces[i][j][2] += step_height*i
    # Create tile
    tile = Tile(p, indices_tile, tile_size, surfaces, type='s')
    if axis==0:
        tile.update_entry(positioning=LEFT  , height=surfaces[0][0][2], open=True)
        tile.update_entry(positioning=RIGHT , height=surfaces[-1][0][2], open=True)
        tile.update_entry(positioning=BOTTOM, height=p[2], open=False)
        tile.update_entry(positioning=TOP   , height=p[2], open=False)
    else:
        tile.update_entry(positioning=BOTTOM, height=surfaces[0][0][2], open=True)
        tile.update_entry(positioning=TOP   , height=surfaces[-1][0][2], open=True)
        tile.update_entry(positioning=LEFT  , height=p[2], open=False)
        tile.update_entry(positioning=RIGHT , height=p[2], open=False)
    return tile

# This will create one pylon on the side of the movement axis (one on the left and one on the right).
# The init is possible on this tile, we have to check where is the obstacle.
# oriented_axis_x : obstacle will be on the right or left of the x-axis if True, y-axis if False.
# If we force an entry, the obstacle can not be at this entry. 
#                       Ex : entry_fooced = [ 0, 1]    => means that the obstacle can not be on the top    (Y axis)
#                            entry_fooced = [ 0,-1] => means that the obstacle can not be on the bottom    (Y axis)
#                            entry_fooced = [ 1, 0]  => means that the obstacle can not be on the right    (X axis)
#                            entry_fooced = [-1, 0]   => means that the obstacle can not be on the left    (X axis)
def tile_obstacle_gen(p, indices_tile, tile_size=[2.0,2.0], positioning_obstacle=BOTTOM, distance_axis_to_obstacle=[0.3,0.5], bounds_height_obstacle=[0.7,2.0]):
    surfaces = tile_flat_gen(p, indices_tile, tile_size=tile_size).surfaces
    # Obstacle on left or right
    pos = p[::][0:2]
    distance_middle_to_obstacle = round(uniform(distance_axis_to_obstacle[0], distance_axis_to_obstacle[1]),2) # PARAMETERS TO TUNE
    # Axis
    if positioning_obstacle==RIGHT or positioning_obstacle==LEFT:
        axis = 0 # Obstacle on the X axis
    else:
        axis = 1 # Obstacle on the Y axis
    # Set width and length of obstacle
    size_obstacle = [ round(uniform(0.15, tile_size[0]/4.0),2), round(uniform(0.15, tile_size[1]/3.0),2) ]  # PARAMETERS TO TUNE
    # Get middle position of obstacle
    height_obstacle  = round(uniform(bounds_height_obstacle[0],bounds_height_obstacle[1]),2)
    if positioning_obstacle==RIGHT or positioning_obstacle==TOP:
        pos[axis] += (distance_middle_to_obstacle+size_obstacle[axis]/2.)
    else :
        pos[axis] -= (distance_middle_to_obstacle+size_obstacle[axis]/2.)
    # Move the obstacle in a corner randomly
    translation_corner = tile_size[axis-1]/4.
    if round(uniform(0.,1.),3)<0.50:
        translation_corner = -translation_corner
    pos[axis-1] += translation_corner
    pos.append(p[2])
    # Create obstacle
    pos[2] += height_obstacle/2.
    surfaces += cube_gen(pos, size_x=size_obstacle[0]/2., size_y=size_obstacle[1]/2., size_z=height_obstacle/2.)
    # Create tile
    tile = Tile(p, indices_tile, tile_size, surfaces, type='o')
    # Get init bounds
    mid_init = 0.   # will be updated
    length_free_space = 0. # will be updated
    if positioning_obstacle==RIGHT:
        pos_limit_obstacle = p[0]
        pos_limit_obstacle += distance_middle_to_obstacle # Obstacle right
        pos_limit_tile = p[0]-tile_size[0]/2. # Limit left
        length_free_space = pos_limit_obstacle - pos_limit_tile
        mid_init = pos_limit_obstacle - length_free_space/2.
        # Set bounds init
        tile.bounds_init[0] = [ mid_init-length_free_space/4., mid_init+length_free_space/4.]
    elif positioning_obstacle==LEFT:
        pos_limit_obstacle = p[0]
        pos_limit_obstacle -= distance_middle_to_obstacle # Obstacle left
        pos_limit_tile = p[0]+tile_size[0]/2. # Limit right
        length_free_space =  pos_limit_tile - pos_limit_obstacle
        mid_init = pos_limit_obstacle + length_free_space/2.
        # Set bounds init
        tile.bounds_init[0] = [ mid_init-length_free_space/4., mid_init+length_free_space/4.]
    elif positioning_obstacle==TOP:
        pos_limit_obstacle = p[1]
        pos_limit_obstacle += distance_middle_to_obstacle # Obstacle top
        pos_limit_tile = p[1]-tile_size[1]/2. # Limit bot
        length_free_space = pos_limit_obstacle - pos_limit_tile
        mid_init = pos_limit_obstacle - length_free_space/2.
        # Set bounds init
        tile.bounds_init[1] = [ mid_init-length_free_space/4., mid_init+length_free_space/4.]
    else: # BOTTOM
        pos_limit_obstacle = p[1]
        pos_limit_obstacle -= distance_middle_to_obstacle # Obstacle bottom
        pos_limit_tile = p[1]+tile_size[1]/2. # Limit top
        length_free_space =  pos_limit_tile - pos_limit_obstacle
        mid_init = pos_limit_obstacle + length_free_space/2.
        # Set bounds init
        tile.bounds_init[1] = [ mid_init-length_free_space/4., mid_init+length_free_space/4. ]
    if axis==0:
        tile.bounds_init[1][0] -= translation_corner
        tile.bounds_init[1][1] -= translation_corner
    else:
        tile.bounds_init[0][0] -= translation_corner
        tile.bounds_init[0][1] -= translation_corner
    """
    print("Tile in : ",p)
    print("Create cube in : ",pos)
    print("Positioning : ",NAME_POSITIONING[positioning_obstacle])
    print("Distance mid to obstacle : ",distance_middle_to_obstacle)
    print("Length free space : ",length_free_space)
    print("Mid init : ",mid_init)
    input("Obstacle tile created...")
    """
    return tile


# TEST TILES
def create_tiles_test():
    scene = []
    tiles = []
    p = np.array([0.,0.,0.])
    tile_step = np.array([2.,0.,0.])
    # flat
    for i in range(0,2):
        print("height : ",p[2])
        tile = tile_flat_gen(p.tolist(), indices_tile=[0,0])
        tiles.append(tile)
        scene += tile.surfaces
        # Rubbles
        p += tile_step
        p[2] = tiles[-1].entries[RIGHT].height
        print("height : ",p[2])
        tile = tile_rubbles_gen(p.tolist(), indices_tile=[1,0])
        tiles.append(tile)
        scene += tile.surfaces
        # bridge
        p += tile_step
        p[2] = tiles[-1].entries[RIGHT].height
        print("height : ",p[2])
        tile = tile_bridge_gen(p.tolist(), indices_tile=[2,0], left_to_right=True, height_LR_or_BT=[p[2],p[2]+0.3])
        tiles.append(tile)
        scene += tile.surfaces
        # obstacle
        p += tile_step
        p[2] = tiles[-1].entries[RIGHT].height
        print("height : ",p[2])
        tile = tile_obstacle_gen(p.tolist(), indices_tile=[3,0])
        tiles.append(tile)
        scene += tile.surfaces
        # stairs
        p += tile_step
        p[2] = tiles[-1].entries[RIGHT].height
        print("height : ",p[2])
        tile = tile_stairs_gen(p.tolist(), indices_tile=[4,0])
        print("STAIRS=========================")
        for entry in tile.entries:
            print(NAME_POSITIONING[entry.positioning_previous_tile]," at ",entry.height)
        tiles.append(tile)
        scene += tile.surfaces
        p += tile_step
        p[2] = tiles[-1].entries[RIGHT].height
    return scene


# ========================================================================================
# ================================== CONNECT TILES =======================================
# ========================================================================================

"""
RECALL OF AVAILABLE FUNCTIONS :
    - tile_flat_gen     (p, indices_tile, tile_size=[2.0,2.0])
    - tile_rubbles_gen  (p, indices_tile, tile_size=[2.0,2.0], bound_h=[0.25,0.35], bound_w=[0.25,0.35])
    - tile_bridge_gen   (p, indices_tile, tile_size=[2.0,2.0], left_to_right=True, width_ramp=0.30)
    - tile_stairs_gen(p, indices_tile, tile_size=[2.0,2.0], positioniong_start=LEFT, height_start=0., height_arrival=1., width_step=[0.2,0.4], length_step=[0.6,1.0])
    - tile_obstacle_gen (p, indices_tile, tile_size=[2.0,2.0], positioning_obstacle=BOTTOM, distance_axis_to_obstacle=[0.3,0.5])
"""

class Grid_terrain:
    def __init__(self, p_start=[0.,0.,0.], tile_size=[2.,2.], divison_grid=[10,10]):
        self.p_start = p_start
        self.divison_grid = divison_grid
        self.grid = [[None for i in range(0,divison_grid[1])] for j in range(0,divison_grid[0])]
        self.links = []
        self.tile_size = tile_size
        self.number_priorities = 5
        self.lists_tiles_to_set_priority = [ [] for i in range(0,self.number_priorities) ]
        self.MAX_CLIMB_BRIDGE = 0.25
        self.MAX_CLIMB_STAIRS = 1.0
        self.force_init_tile = False
        return

    def init_start_tile(self, indices_tile=[0,0], is_obstacle=False):
        p = (np.array(self.p_start)+np.array([self.tile_size[0]*indices_tile[0],self.tile_size[1]*indices_tile[1],0.])).tolist()
        is_obstacle = False
        if is_obstacle:
            tile = tile_obstacle_gen(p=p, indices_tile=indices_tile, tile_size=self.tile_size, distance_axis_to_obstacle=[0.3,0.5])
        else:
            tile = tile_flat_gen(p=p, indices_tile=indices_tile, tile_size=self.tile_size)
        self.set_grid(tile, indices_tile)
        self.add_tiles_to_set(tile)
        return

    def set_grid(self, tile, tile_indices):
        print("Tile indices : ",tile_indices)
        self.grid[tile_indices[0]][tile_indices[1]] = tile
        return

    def get_tile(self, tile_indices):
        return self.grid[tile_indices[0]][tile_indices[1]]

    def get_all_surfaces(self):
        surfaces = []
        for row in self.grid:
            for tile in row:
                if tile is not None:
                    surfaces += tile.surfaces
        return surfaces

    def check_grid_full(self):
        list_indices_none = []
        for i in range(0,len(self.grid)):
            for j in range(0,len(self.grid[i])):
                if self.grid[i][j] is None:
                    list_indices_none.append([i,j])
        is_full = (len(list_indices_none)==0)
        return is_full, list_indices_none

    def add_tiles_to_set(self, tile, tile_to_set=None):
        priority = -1
        if tile.type=='f':      # Flat
            priority = 3
        elif tile.type=='o':    # Obstacle
            priority = 3
        elif tile.type=='r':    # Rubbles
            priority = 2
        elif tile.type=='b':    # Bridge
            priority = 1
        elif tile.type=='s':    # Stairs
            priority = 0
        list_tiles_first = self.lists_tiles_to_set_priority[priority] # Set a tile first just on the other side of non-init-tile (Ex : stairs with entry on left => set tile on the right)
        list_tiles_second = self.lists_tiles_to_set_priority[3] # the other tiles will have a default priority
        entry = None
        if not tile.is_init_tile():
            entry = tile_to_set.entry
        # Check tiles to set
        print("Tile of type : ",tile.type)
        if entry is not None:
            print("with entry : ",tile_to_set.entry)
        # bot
        i = tile.indices[0]
        j = tile.indices[1]-1
        if self.are_indices_in_grid([i,j]) and self.get_tile([i,j]) is None:
            if tile.entries[BOTTOM].open:
                tile_to_set = Tile_to_set([i,j], previous_type=tile.type, entry=tile.entries[BOTTOM])
                if entry is not None:
                    if entry.positioning_previous_tile==TOP:
                        list_tiles_first.append(tile_to_set)
                    else:
                        list_tiles_second.append(tile_to_set)
                else:
                    list_tiles_second.append(tile_to_set)
        # top
        i = tile.indices[0]
        j = tile.indices[1]+1
        if self.are_indices_in_grid([i,j]) and self.get_tile([i,j]) is None:
            if tile.entries[TOP].open:
                tile_to_set = Tile_to_set([i,j], previous_type=tile.type, entry=tile.entries[TOP])
                if entry is not None:
                    if entry.positioning_previous_tile==BOTTOM:
                        list_tiles_first.append(tile_to_set)
                    else:
                        list_tiles_second.append(tile_to_set)
                else:
                    list_tiles_second.append(tile_to_set)
        # left
        i = tile.indices[0]-1
        j = tile.indices[1]
        if self.are_indices_in_grid([i,j]) and self.get_tile([i,j]) is None:
            if tile.entries[LEFT].open:
                tile_to_set = Tile_to_set([i,j], previous_type=tile.type, entry=tile.entries[LEFT])
                if entry is not None:
                    if entry.positioning_previous_tile==RIGHT:
                        list_tiles_first.append(tile_to_set)
                    else:
                        list_tiles_second.append(tile_to_set)
                else:
                    list_tiles_second.append(tile_to_set)
        # right
        i = tile.indices[0]+1
        j = tile.indices[1]
        if self.are_indices_in_grid([i,j]) and self.get_tile([i,j]) is None:
            if tile.entries[RIGHT].open:
                tile_to_set = Tile_to_set([i,j], previous_type=tile.type, entry=tile.entries[RIGHT])
                if entry is not None:
                    if entry.positioning_previous_tile==LEFT:
                        list_tiles_first.append(tile_to_set)
                    else:
                        list_tiles_second.append(tile_to_set)
                else:
                    list_tiles_second.append(tile_to_set)
        return

    def fill_grid_random(self, fill_none_randomly=True):
        # Fill grid => Create a path
        id_tile_path = 0
        while (not self.are_lists_to_set_empty()):
            print("*********** Tile on path number : ",id_tile_path," **************")
            self.create_next_tile()
            id_tile_path+=1
            #grid_terrain.plot_surfaces()
            #input("Create next tile ...")
        # Fill the rest of the grid with random things
        if fill_none_randomly:
            print("**************** All tiles have been filled, we fill the others that are None *************")
            is_full, list_indices_none = self.check_grid_full()
            number_tiles_none = 0
            if not is_full:
                list_positioning = [BOTTOM, TOP, LEFT, RIGHT]
                for indices in list_indices_none:
                    # Get tiles around and pick one randomly which is not None
                    list_tiles_around = self.get_tiles_around(indices)
                    if len(list_tiles_around)!=0:
                        # We found a random tile around not None
                        # Pick a random one
                        tile = list_tiles_around[ randint(0,len(list_tiles_around)-1) ]
                        positioning_vector = np.array(indices)-np.array(tile.indices)
                        positioning=0
                        while((VEC_POSITIONING[positioning]!=positioning_vector).all()) and positioning<4:
                            positioning += 1
                        if positioning==4:
                            print("ERROR, positioning value at 4 should not be possible")
                        else:
                            entry = Entry(tile.indices, positioning, height=tile.p[2], open=False)
                            self.set_grid(self.get_random_tile_close(indices, entry), tile_indices=indices)
        return

    def are_lists_to_set_empty(self):
        are_emtpy = True
        for i in range(0,len(self.lists_tiles_to_set_priority)):
            if len(self.lists_tiles_to_set_priority[i])!=0:
                are_emtpy = False
                break
        return are_emtpy

    def create_next_tile(self):
        tile_to_set = self.get_random_tile_to_set()
        # INIT TILES  : Flat or Obstacle
        # NON-INIT TILES : Stairs, Bridge, Rubbles
        # What we do :
        #       - prev is init tile     => Create non-init tile
        #       - prev is non-init tile => Create init tile
        tile = None
        probability_init_tile = 0.2
        random_number = round(uniform(0.,1.),3)
        if not self.force_init_tile and tile_to_set.is_previous_init_tile() and random_number>probability_init_tile:
            tile = self.get_random_non_init_tile(tile_to_set.indices, tile_to_set.entry)
        else:
            tile = self.get_random_init_tile(tile_to_set.indices, tile_to_set.entry)
        self.add_tiles_to_set(tile, tile_to_set=tile_to_set)
        self.set_grid(tile, tile_to_set.indices)
        #self.plot_surfaces()
        return

    # Get Flat ground or Obstacle
    def get_random_init_tile(self, indices_tile, entry):
        tile = None
        p = self.get_pos_2D(indices_tile)
        p.append(entry.height)
        # if randint(0, 1)==0 or self.force_init_tile:
        # Flat
        print("Create flat ground in ",indices_tile)
        tile = tile_flat_gen(p, indices_tile, tile_size=self.tile_size)
        # else:
        #     # Obstacle
        #     print("Create obstacle tile in ",indices_tile)
        #     entry_position = entry.positioning_previous_tile
        #     list = [BOTTOM, TOP, LEFT, RIGHT]
        #     list.remove(entry_position)
        #     positioning_obstacle = list[ randint(0,2) ]
        #     tile = tile_obstacle_gen(p, indices_tile, tile_size=self.tile_size, positioning_obstacle=positioning_obstacle, distance_axis_to_obstacle=[0.3,0.5])
        return tile

    # Get Stairs, Rubbles or Bridge
    def get_random_non_init_tile(self, indices_tile, entry):
        tile = None
        p = self.get_pos_2D(indices_tile)
        p.append(entry.height)
        # Check if on the other side, there is already a tile
        positioning_previous_tile = entry.positioning_previous_tile
        vec_positioning_previous_tile = VEC_POSITIONING[positioning_previous_tile]
        indices_tile_prev = (np.array(indices_tile)+np.array(vec_positioning_previous_tile)).tolist()
        indices_tile_other_side = (np.array(indices_tile)+np.array(vec_positioning_previous_tile)*-1).tolist()
        height_other_side = None
        # Check if tiles exists
        if self.are_indices_in_grid(indices_tile_other_side):
            # Check if is not None
            tile_other_side = self.get_tile(indices_tile_other_side)
            if tile_other_side is not None:
                if tile_other_side.type=='f' or tile_other_side.type=='o':
                    height_other_side = tile_other_side.p[2]
        # if height < 15 cm => Bridge or rubbles
        #    height < 25 cm => Bridge or Stairs
        #    height < 1m    => Stairs
        # if too high or tile other side is None => Bridge, Stairs or Rubbles
        if height_other_side is not None:
            # If the change in height is inferior to 15 centimers => bridge
            if abs(entry.height-height_other_side) < 0.15:
                #random_choice = randint(1,2)
                random_choice=1
            elif abs(entry.height-height_other_side)<self.MAX_CLIMB_BRIDGE:
                # Lower than 25 centimers => bridge or stairs
                random_choice = randint(0,1)
            elif height_other_side<self.MAX_CLIMB_STAIRS:
                # Lower than MAX_CLIMB_STAIRS meters => stairs
                random_choice = 0
            else:
                # Tile on the other side is too high or low, so we forget it and get a random tile.
                height_other_side = None
                random_choice = randint(0, 2)
        else:
            # If there is nothing on the other side
            percentage_chance_stairs = 0.55
            percentage_chance_bridge = 0.30
            number = round(uniform(0.,1.),3)
            if number<percentage_chance_stairs:
                # Stairs
                random_choice = 0
            elif number <percentage_chance_stairs+percentage_chance_bridge:
                # Bridge
                random_choice = 1
            else:
                # Rubbles => This is quite heavy to load (Many triangles)
                random_choice = 2
        # Create the tile of chosen type
        if random_choice==0:# and False:
            # Stairs
            # We would like the scene to stay with a Z around 0.0 (Not go to high or low)
            # Z<-treshold_height_wanted => 100% chance to go up   | from -treshold_height_wanted to 0 => chances 100% to 50%
            # Z>treshold_height_wanted  => 100% chance to go down | from 0 to treshold_height_wanted  => chances 50% to 100%
            treshold_height_wanted = 1.2
            bounds_height_to_climb = [0.4,1.0] # positive
            height_arrival = None
            if height_other_side is None:
                if p[2]>=0:
                    probability_go_up = min(p[2],1.2) / treshold_height_wanted
                    probability_go_down = 1.0-probability_go_up
                else:
                    probability_go_up = -(max(p[2],-1.2) / treshold_height_wanted)
                    probability_go_down = 1.0-probability_go_up
                # Pick random height to climb according to probabilities
                random_float = round(uniform(0.,1.),3)
                if random_float < probability_go_up:
                    # We go up
                    height_to_climb = round(uniform(bounds_height_to_climb[0],bounds_height_to_climb[1]),2)
                else:
                    # We go down
                    height_to_climb = -round(uniform(bounds_height_to_climb[0],bounds_height_to_climb[1]),2)
                height_arrival = entry.height+height_to_climb
            else:
                height_arrival = height_other_side
            print("Create stairs in ",indices_tile," starting ",NAME_POSITIONING[entry.positioning_previous_tile],
                  " at h=",entry.height," and finishing at h=",height_arrival)
            min_tile_size = min(self.tile_size[0],self.tile_size[1])
            tile = tile_stairs_gen(p, indices_tile, tile_size=self.tile_size, positioniong_start=entry.positioning_previous_tile, 
                                   width_step=[0.2,0.4], length_step=[min_tile_size-min_tile_size/6,min_tile_size], height_start=entry.height, height_arrival=height_arrival)
        elif random_choice==1:# and False:
            # Bridge
            left_to_right = (entry.positioning_previous_tile==LEFT or entry.positioning_previous_tile==RIGHT)
            proba_large_bridge = round(uniform(0.0,1.0),2)
            if proba_large_bridge<0.30:
                # Like a large flat ground
                min_tile_size = min( self.tile_size[0], self.tile_size[1] )
                width_ramp = round(uniform(min_tile_size-0.20,min_tile_size),2)
            else:
                width_ramp = round(uniform(0.30,0.45),2)
            print("Create bridge in ",indices_tile," oriented left_to_right=",left_to_right," and of width=",width_ramp," at height=",entry.height)
            if height_other_side is None:
                tile = tile_bridge_gen(p, indices_tile, tile_size=self.tile_size, left_to_right=left_to_right, width_ramp=width_ramp)
            else:
                if entry.positioning_previous_tile==RIGHT or entry.positioning_previous_tile==TOP:
                    height_LR_or_BOT = [entry.height, height_other_side]
                else:
                    height_LR_or_BOT = [height_other_side, entry.height]
                tile = tile_bridge_gen(p, indices_tile, tile_size=self.tile_size, left_to_right=left_to_right, width_ramp=width_ramp, height_LR_or_BT=height_LR_or_BOT)
        elif random_choice==2:
            # Rubbles
            print("Create rubbles in ",indices_tile)
            tile = tile_rubbles_gen(p, indices_tile, tile_size=self.tile_size, bound_h=[0.35,0.45], bound_w=[0.35,0.45])
        return tile

    def get_random_tile_to_set(self):
        # Should we have some priorities ?
        # Maybe setting first the 'f' or 'o' tile after stairs/bridge/rubbles
        indices = [0,0]
        tile_to_set = None
        value_priority = 0
        while((tile_to_set is None) and (value_priority<self.number_priorities)):
            number_tiles_to_set = len(self.lists_tiles_to_set_priority[value_priority])
            if (number_tiles_to_set>0):
                #tile_to_set = self.lists_tiles_to_set_priority[value_priority][ randint(0,number_tiles_to_set-1) ]
                tile_to_set = self.lists_tiles_to_set_priority[value_priority][ 0 ]
                print("Tiles to set with priority ",value_priority," and previous type ",tile_to_set.previous_type)
                # Remove this elementfrom list
                self.lists_tiles_to_set_priority[value_priority].remove(tile_to_set)
            value_priority += 1
        if tile_to_set is None:
            print("Error, tile_to_set should not be None in get_random_tile_to_set")
        return tile_to_set

    def get_random_tile_close(self, indices_tile, entry):
        # 2 init and 3 non-init tiles.
        random_choice = randint(0, 4)
        entry_close = Entry(indices_tile, positioning_next_tile=LEFT, open=False) # positioning does not matter here. We close all entry of this tile.
        if random_choice<2:
            # init
            tile = self.get_random_init_tile(indices_tile, entry)
        else:
            # non-init
            tile = self.get_random_non_init_tile(indices_tile, entry)
        tile.entries = [ entry_close,entry_close,entry_close,entry_close ]
        return tile

    def get_pos_2D(self, indices_tile):
        #print("Indices tiles : ",indices_tile," pos : ",(np.array(self.p_start[0:2])+ np.array(self.tile_size)/2.0 + np.array(self.tile_size)*np.array(indices_tile)).tolist())
        return (np.array(self.p_start[0:2]) + np.array(self.tile_size)*np.array(indices_tile)).tolist()

    def get_tiles_around(self, indices_tile):
        list_tiles = []
        list_positioning = [BOTTOM, TOP, LEFT, RIGHT]
        for positioning in list_positioning:
            tile_indices_to_test = (np.array(indices_tile)+np.array(VEC_POSITIONING[positioning])).tolist()
            if self.are_indices_in_grid(tile_indices_to_test):
                tile = self.get_tile(tile_indices_to_test)
                if tile is not None:
                    list_tiles.append(tile)
        return list_tiles

    def are_indices_in_grid(self, indices):
        i, j = indices[0], indices[1]
        #print("Indices ",indices," in grid = ",(j>=0 and j<self.divison_grid[1] and i>=0 and i<self.divison_grid[0]))
        return (j>=0 and j<self.divison_grid[1] and i>=0 and i<self.divison_grid[0])

    def plot_surfaces(self):
        surfaces = grid_terrain.get_all_surfaces()
        ax = draw_scene(surfaces)
        plt.show(block=False)
        return

    """
    def print_grid_type(self):
        print("GRID type")
        print("---------")
        for i in range(0,len(self.grid)):
            for j in range(0, len(self.grid[i])):
                print(self.grid[i][j].type, end=',')
            print("")
        print("---------")
        return
    """

    def update_links(self):
        # Go from left to right
        for i in range(0,len(self.grid)):
            for j in range(0,len(self.grid[i])):
                tile = self.grid[i][j]
                if tile is not None:
                    # if the type is an init bloc
                    if tile.type=='f' or tile.type=='o':
                        # ==== Check link : left to right
                        aux_i = i+1
                        if self.are_indices_in_grid([aux_i,j]):
                            tile_next = self.get_tile([aux_i,j])
                            if tile_next is not None:
                                # Check if next tile is init, if yes => Check if the height difference is +-10cm the same => Create a link
                                if tile_next.is_init_tile():
                                    if abs(tile.p[2]-tile_next.p[2])<0.1:
                                        bounds_init = tile.bounds_init
                                        bounds_goal = tile_next.bounds_init
                                        link = Link(bounds_init, bounds_goal, [i,j], [aux_i,j], tile_next.type)
                                        self.links.append(link)
                                else:
                                    # We may have an init tile -> non-init tile -> init tile
                                    # 1- Check if tile other side is init, if yes =>
                                    # 2- Check if tiles [i,j] open RIGHT -> [i+1,j] open LEFT & RIGHT -> [i+2,j] open LEFT
                                    # 3- Check the difference of height between the 3 tiles.
                                    #   3.1- init-rubbles/flat/obstacle-init : Lower than 0.1m => ok
                                    #   3.2- init-bridge/stairs-init : Lower than self.MAX_CLIMB_BRIDGE => ok
                                    #   3.3- init-stairs-init : Lower than self.MAX_CLIMB_STAIRS => ok
                                    aux_i = i+2
                                    if self.are_indices_in_grid([aux_i,j]):
                                        tile_next_next = self.get_tile([aux_i,j])
                                        if tile_next_next is not None:
                                            # condition 1
                                            if tile_next_next.is_init_tile():
                                                # condition 2
                                                if tile.entries[RIGHT].open and tile_next.entries[LEFT].open and tile_next.entries[RIGHT].open and tile_next_next.entries[LEFT].open :
                                                    # Condition 3
                                                    is_linkable = False
                                                    if (tile_next.type=='b' or tile_next.type=='s') and abs(tile.p[2]-tile_next_next.p[2])<self.MAX_CLIMB_BRIDGE:
                                                        is_linkable = True
                                                    elif tile_next.type=='s' and abs(tile.p[2]-tile_next_next.p[2])<self.MAX_CLIMB_STAIRS:
                                                        is_linkable = True
                                                    elif abs(tile.p[2]-tile_next.p[2])<0.1 and abs(tile_next.p[2]-tile_next_next.p[2])<0.1: # 'r'/'f'/'o'
                                                        is_linkable = True
                                                    if is_linkable:
                                                        bounds_init = tile.bounds_init
                                                        bounds_goal = tile_next_next.bounds_init
                                                        link = Link(bounds_init, bounds_goal, [i,j], [aux_i,j], tile_next.type)
                                                        self.links.append(link)
                        # ==== Check link : bottom to top
                        aux_j = j+1
                        if self.are_indices_in_grid([i,aux_j]):
                            tile_next = self.get_tile([i,aux_j])
                            if tile_next is not None:
                                # Check if next tile is init, if yes => Check if the height difference is +-10cm the same => Create a link
                                if tile_next.is_init_tile():
                                    if abs(tile.p[2]-tile_next.p[2])<0.1:
                                        bounds_init = tile.bounds_init
                                        bounds_goal = tile_next.bounds_init
                                        link = Link(bounds_init, bounds_goal, [i,j], [i,aux_j], tile_next.type)
                                        self.links.append(link)
                                else:
                                    # We may have an init tile -> non-init tile -> init tile
                                    # 1- Check if tile other side is init, if yes =>
                                    # 2- Check if tiles [i,j] open TOP -> [i+1,j] open BOT & TOP -> [i+2,j] open BOT
                                    # 3- Check if 
                                    aux_j = j+2
                                    if self.are_indices_in_grid([i,aux_j]):
                                        tile_next_next = self.get_tile([i,aux_j])
                                        if tile_next_next is not None:
                                            # condition 1
                                            if tile_next_next.is_init_tile():
                                                # condition 2
                                                if tile.entries[TOP].open and tile_next.entries[BOTTOM].open and tile_next.entries[TOP].open and tile_next_next.entries[BOTTOM].open :
                                                    # Condition 3
                                                    is_linkable = False
                                                    if (tile_next.type=='b' or tile_next.type=='s') and abs(tile.p[2]-tile_next_next.p[2])<self.MAX_CLIMB_BRIDGE:
                                                        is_linkable = True
                                                    elif tile_next.type=='s' and abs(tile.p[2]-tile_next_next.p[2])<self.MAX_CLIMB_STAIRS:
                                                        is_linkable = True
                                                    elif abs(tile.p[2]-tile_next.p[2])<0.1 and abs(tile_next.p[2]-tile_next_next.p[2])<0.1: # 'r'/'f'/'o'
                                                        is_linkable = True
                                                    if is_linkable:
                                                        bounds_init = tile.bounds_init
                                                        bounds_goal = tile_next_next.bounds_init
                                                        link = Link(bounds_init, bounds_goal, [i,j], [i,aux_j], tile_next.type)
                                                        self.links.append(link)
        return


    def set_test(self, height, start_middle):
        self.divison_grid=[5,5]
        self.grid = [[None for i in range(0,self.divison_grid[1])] for j in range(0,self.divison_grid[0])]
        # Mid
        indices_tile = [2,2]
        p = self.get_pos_2D(indices_tile)
        p.append(0.)
        tile = tile_flat_gen(p=p, indices_tile=indices_tile, tile_size=self.tile_size)
        self.set_grid(tile, indices_tile)
        # BOTTOM
        indices_tile = [2,0]
        p = self.get_pos_2D(indices_tile)
        p.append(height)
        tile = tile_flat_gen(p=p, indices_tile=indices_tile, tile_size=self.tile_size)
        self.set_grid(tile, indices_tile)
        # TOP
        indices_tile = [2,4]
        p = self.get_pos_2D(indices_tile)
        p.append(height)
        tile = tile_flat_gen(p=p, indices_tile=indices_tile, tile_size=self.tile_size)
        self.set_grid(tile, indices_tile)
        # LEFT
        indices_tile = [0,2]
        p = self.get_pos_2D(indices_tile)
        p.append(height)
        tile = tile_flat_gen(p=p, indices_tile=indices_tile, tile_size=self.tile_size)
        self.set_grid(tile, indices_tile)
        # RIGHT
        indices_tile = [4,2]
        p = self.get_pos_2D(indices_tile)
        p.append(height)
        tile = tile_flat_gen(p=p, indices_tile=indices_tile, tile_size=self.tile_size)
        self.set_grid(tile, indices_tile)
        # Tile to set =================
        previous_type = 'f'
        # Bottom
        indices = [2,1]
        if start_middle:
            entry = Entry(indices, BOTTOM, height=0., open=True)
        else:
            entry = Entry(indices, TOP, height=height, open=True)
        tile = self.get_random_non_init_tile(indices, entry)
        self.set_grid(tile, indices)
        # Top
        indices = [2,3]
        if start_middle:
            entry = Entry(indices, TOP, height=0., open=True)
        else:
            entry = Entry(indices, BOTTOM, height=height, open=True)
        tile = self.get_random_non_init_tile(indices, entry)
        self.set_grid(tile, indices)
        # Left
        indices = [1,2]
        if start_middle:
            entry = Entry(indices, LEFT, height=0., open=True)
        else:
            entry = Entry(indices, RIGHT, height=height, open=True)
        tile = self.get_random_non_init_tile(indices, entry)
        self.set_grid(tile, indices)
        # Right
        indices = [3,2]
        if start_middle:
            entry = Entry(indices, RIGHT, height=0., open=True)
        else:
            entry = Entry(indices, LEFT, height=height, open=True)
        tile = self.get_random_non_init_tile(indices, entry)
        self.set_grid(tile, indices)
        return

    def print_links(self):
        print("**** Print links ****")
        for link in self.links:
            print(" - ",link)
        return

# ========================================================================================
# ====================================== TOOLS ===========================================
# ========================================================================================

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
        
def draw_scene(surfaces, ax = None, points_scatter = None):
    colors = ['r','g','b','m','y','c']
    color_id = 0
    if ax is None:        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for surface in surfaces:
        plot_surface(np.array(surface).T, ax)
    if points_scatter is not None:
        for point in points_scatter:
            plt.scatter(np.array(point), c='b', s=30.0, linewidths=0.0000001, alpha=1.0)
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

def save_links(filename, grid_terrain):
    # We want something like : list_links = [ [Z_0,Z_1], [Z_0, Z_2], ... ]
    list_links = []
    for link in grid_terrain.links:
        list_links.append( [link.Z_0, link.Z_1] ) # append [bounds_zone_0, bounds_zone_1]
    # Write this list in file
    f = open(filename,'w')
    f.write('LIST_LINKS = '+str(list_links))
    f.close()
    return


# ========================================================================================
# ======================================= MAIN ===========================================
# ======================================================================================== 

"""
if __name__ == '__main__':
    all_surfaces = scene_gen()
    ax = draw_scene(all_surfaces)
    plt.show()
    fileName = sys.argv[1]
    save_obj("obj/"+fileName+".obj", all_surfaces)
    save_pb("pb/"+fileName, all_surfaces)
"""

"""
if __name__ == '__main__':
    all_surfaces = create_tiles_test()
    #print("All surfaces : ", all_surfaces)
    ax = draw_scene(all_surfaces)
    plt.show()
    fileName = sys.argv[1]
    save_obj("obj/"+fileName+".obj", all_surfaces)
    #save_pb("pb/"+fileName, all_surfaces)


if __name__ == '__main__':
    grid_terrain = Grid_terrain()
    grid_terrain.set_test(height=-0.8, start_middle=False)

"""

if __name__ == '__main__':
    grid_terrain = Grid_terrain(divison_grid=[10,10]) # Pick a size for your grid, here it's 5x5
    grid_terrain.init_start_tile([1,1]) # Choose where to put the first flat tile from where to start
    grid_terrain.force_init_tile = False # Fill the terrain with only flat tile
    grid_terrain.fill_grid_random(fill_none_randomly=True) # Fill all the grid. If some tiles are none at the end, if fill_none_randomly is True, we fille them.
    #grid_terrain.print_grid_type()
    grid_terrain.update_links() # Update all links (init and goal zones that are possible)
    #grid_terrain.print_links()
    grid_terrain.plot_surfaces() # plot surfaces
    # Save or not the terrain and links
    if True:
        surfaces = grid_terrain.get_all_surfaces()
        file_name = sys.argv[1]
        #input("...")
        save_links("links/"+file_name+"_links", grid_terrain)
        save_obj("obj/"+file_name+".obj", grid_terrain.get_all_surfaces())
        save_pb("pb/"+file_name, surfaces)
    
