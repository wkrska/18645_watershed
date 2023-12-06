SCL = scl enable devtoolset-11
CC = gcc
CFLAGS = -mavx -mavx2 -mfma -fopenmp -lm -O1 -std=c99 -w -fno-align-functions

default: all run assemble

dev: scl run assemble

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: 
	$(CC) $(CFLAGS) test.c -o test -march=native

scl:
	$(SCL) '$(CC) $(CFLAGS) test.c morph_kernel.c -o test -march=native'

run:
	./test

assemble:
	objdump -s -d -f --source test > test.S

clean:
	rm -f *.x *~ *.o
