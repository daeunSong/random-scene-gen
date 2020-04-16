#!/bin/bash         

python ./random_scene_gen.py ./$1 &
assimp export ./$1.obj ./$1.stl

