## ---------------------------------------------------------------------------
##      M a k e f i l e   f o r   i s i n g 2 d.c	    f i l e
## ---------------------------------------------------------------------------
## This file produces one executable: "ising2d" and one object file named
## as "ising2d.o". 


# CC	=	gcc-11#	-fopenmp#	-lmpi#	-showme
CC	=	mpicc	-fopenmp#	-fopenmp#	-lmpi#	-showme
INC_DIR = /Users/jailsonoliveira/Desktop/Monte_Carlo/IsingModelFerromag/Ising2DFerromag/
CFLAGS	=	-std=c99	-w	-Wall	-I$(INC_DIR)
# CFLAGS	=	-std=c99	-pedantic	-Wall	-I$(INC_DIR)
# DEPS = forces.h	#Unused for now (and INC_DIR)
OBJECTS	=	ising2d.o

all:	ising2d

ising2d.o:	ising2d.c
	$(CC)	$(CFLAGS)	-c	ising2d.c	-lm

ising2d:	$(OBJECTS)
	$(CC)	$(OBJECTS)	-o	ising2d	-lm

clean:
	rm	-f	*.o	ising2d