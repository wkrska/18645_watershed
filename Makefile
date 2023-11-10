SCL = scl enable devtoolset-11
CC = gcc
CFLAGS = -mavx -mavx2 -mfma -lm -O3 -std=c99 -w

default: all run assemble

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: 
	$(CC) $(CFLAGS) test.c  -o test -march=native
run:
	./test

assemble:
	objdump -s -d -f --source test > test.S

clean:
	rm -f *.x *~ *.o
