SCL = scl enable devtoolset-9
# CC = gcc
CC = mpicc -n 4
CFLAGS = -mavx -mavx2 -mfma -fopenmp -lm -O1 -std=c99 -w #-fno-align-functions

default: all run assemble

dev: scl run assemble

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: 
	$(CC) $(CFLAGS) test.c -o test.x -march=native

scl:
	$(SCL) '$(CC) $(CFLAGS) test.c -o test.x -march=native'

run:
	./test.x

assemble:
	objdump -s -d -f --source test.x > test.S

clean:
	rm -f *.x *~ *.S
