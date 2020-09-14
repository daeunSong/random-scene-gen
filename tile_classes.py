import numpy as np

BOTTOM  = 0
TOP     = 1
RIGHT   = 2
LEFT    = 3

VEC_POSITIONING = [ [0,-1], [0,1], [1,0], [-1,0] ]
NAME_POSITIONING = [ 'BOTTOM', 'TOP', 'RIGHT', 'LEFT' ]

# Each tile has 4 entry of different heights which can be open or close
# [ 0, 1]  -> top
# [ 0,-1]  -> bottom
# [ 1, 0]  -> right
# [-1, 0]  -> left
class Entry:
    def __init__(self, indices_actual_tile, positioning_next_tile, height=-1.0,open=True):
        self.indices_next_tile = (np.array(indices_actual_tile)+np.array(VEC_POSITIONING[positioning_next_tile])).tolist()
        self.height = height
        self.open = open
        self.positioning_previous_tile = None
        # If the tile is at the right of previous tile, then her entry is on the left, etc...
        if positioning_next_tile==LEFT:
            self.positioning_previous_tile = RIGHT
        elif positioning_next_tile==RIGHT:
            self.positioning_previous_tile = LEFT
        elif positioning_next_tile==TOP:
            self.positioning_previous_tile = BOTTOM
        else: # BOTTOM
            self.positioning_previous_tile = TOP
        return
    def __str__(self):
        my_string = "Entry : indices_next_tile("+str(self.indices_next_tile)+") positioning_previous_tile("+str(self.positioning_previous_tile)+")\n"
        my_string+= "        height("+str(self.height)+") open("+str(self.open)+")"
        return my_string


class Tile:
    def __init__(self, p, indices_tile, tile_size, surfaces, type='f'):
        self.type = type
        self.p = p
        self.init_possible = (type=='f' or type=='o') # True on flat ground and obstacle
        self.bounds_init = [ [p[0]-tile_size[0]/4.,p[0]+tile_size[0]/4.], [p[1]-tile_size[1]/4.,p[1]+tile_size[1]/4.], [p[2],p[2]+0.02] ] # [ bounds_x, bounds_y, bounds_z ]
        self.tile_size = tile_size
        self.surfaces = surfaces
        self.indices = indices_tile
        # By default : entries are open and set to the same height than the pos of tile.
        self.entries = [ Entry(self.indices, BOTTOM, height=p[2], open=True),
                         Entry(self.indices, TOP   , height=p[2], open=True),
                         Entry(self.indices, RIGHT , height=p[2], open=True),
                         Entry(self.indices, LEFT  , height=p[2], open=True)  ]
        return
    def update_entry(self, positioning, height=0.0, open=True):
        self.entries[positioning].height = height
        self.entries[positioning].open = open
        return
    def is_init_tile(self):
        return self.type=='f' or self.type=='o'


class Tile_to_set:
    def __init__(self, indices_tile, previous_type, entry):
        self.indices = indices_tile
        self.previous_type = previous_type
        self.entry = entry
        return
    def is_previous_init_tile(self):
        return self.previous_type=='f' or self.previous_type=='o' # Flat or Obstacle


class Link:
    def __init__(self, bounds_init, bounds_goal, indices_init, indices_goal, type):
        self.Z_0 = bounds_init
        self.Z_1 = bounds_goal
        self.indices_Z_0 = indices_init
        self.indices_Z_1 = indices_goal
        self.type = type
        return

    def __str__(self):
        my_string = "Link of type "+str(self.type)+" :\n"
        my_string+= "   Z_0("+str(self.Z_0)+" of indices("+str(self.indices_Z_0)+")\n"
        my_string+= "   Z_1("+str(self.Z_1)+" of indices("+str(self.indices_Z_1)+")"
        return my_string