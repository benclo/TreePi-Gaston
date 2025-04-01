# Compiler and flags
GCC=g++ -O3

# List all source files
ALLFILES=TreePi.cpp BTreePlus.cpp closeleg.cpp database.cpp Graph.cpp graphstate.cpp legoccurrence.cpp path.cpp patterngraph.cpp patterntree.cpp gaston.cpp

# List all object files corresponding to the source files
OBJS=TreePi.o BTreePlus.o closeleg.o database.o Graph.o graphstate.o legoccurrence.o path.o patterngraph.o patterntree.o gaston.o

# Define the target for the final executable
TreePi: $(OBJS)
	$(GCC) -o TreePi $(OBJS)

# Clean up the build
clean:
	rm $(OBJS); rm TreePi

# Compile individual object files
TreePi.o: TreePi.cpp BTreePlus.h closeleg.h database.h Graph.h graphstate.h legoccurrence.h misc.h path.h patterngraph.h patterntree.h gaston.h
	$(GCC) -c TreePi.cpp

BTreePlus.o: BTreePlus.cpp BTreePlus.h
	$(GCC) -c BTreePlus.cpp

closeleg.o: closeleg.cpp closeleg.h misc.h legoccurrence.h
	$(GCC) -c closeleg.cpp

database.o: database.cpp database.h
	$(GCC) -c database.cpp

Graph.o: Graph.cpp Graph.h database.h
	$(GCC) -c Graph.cpp

graphstate.o: graphstate.cpp graphstate.h database.h misc.h
	$(GCC) -c graphstate.cpp

legoccurrence.o: legoccurrence.cpp legoccurrence.h closeleg.h database.h graphstate.h
	$(GCC) -c legoccurrence.cpp

path.o: path.cpp path.h patterntree.h patterngraph.h graphstate.h
	$(GCC) -c path.cpp

patterngraph.o: patterngraph.cpp patterngraph.h graphstate.h
	$(GCC) -c patterngraph.cpp

patterntree.o: patterntree.cpp patterntree.h patterngraph.h graphstate.h
	$(GCC) -c patterntree.cpp

gaston.o: gaston.cpp gaston.h database.h misc.h
	$(GCC) -c gaston.cpp
