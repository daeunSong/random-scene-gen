#!/bin/bash         

python ./random_scene_gen.py ./$1
assimp export obj/$1.obj stl/$1.stl
blender --background --python script.py -- $1

