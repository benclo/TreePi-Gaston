#!/bin/bash
#SBATCH -p main      # Queue name
#SBATCH --mem=188000

# Compiler and flags
GCC="g++ -O3"

# List all source files
SRC_FILES="TreePi.cpp BTreePlus.cpp closeleg.cpp database.cpp Graph.cpp graphstate.cpp legoccurrence.cpp path.cpp patterngraph.cpp patterntree.cpp gaston.cpp"

# List all object files
OBJ_FILES="TreePi.o BTreePlus.o closeleg.o database.o Graph.o graphstate.o legoccurrence.o path.o patterngraph.o patterntree.o gaston.o"

# Compile individual object files
for src in $SRC_FILES; do
    obj="${src%.cpp}.o"
    $GCC -c $src -o $obj
done

$GCC -o TreePi $OBJ_FILES

./TreePi
