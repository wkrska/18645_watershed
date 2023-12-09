SCL = scl enable devtoolset-9
CC = gcc
CFLAGS = -mavx -mavx2 -mfma -fopenmp -lm -O3 -std=c99 -w #-fno-align-functions

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

batch:
	$(CC) $(CFLAGS) -DROWS=4    -DCOLS=32 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test4.txt
	$(CC) $(CFLAGS) -DROWS=8    -DCOLS=32 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test8.txt
	$(CC) $(CFLAGS) -DROWS=16   -DCOLS=32 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test16.txt
	$(CC) $(CFLAGS) -DROWS=32   -DCOLS=32 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test32.txt
	$(CC) $(CFLAGS) -DROWS=64   -DCOLS=32 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test64.txt
	$(CC) $(CFLAGS) -DROWS=128  -DCOLS=32 -DRUNS=100000000 test.c -o test.x -march=native
	./test.x -> test128.txt
	$(CC) $(CFLAGS) -DROWS=256  -DCOLS=32 -DRUNS=100000000 test.c -o test.x -march=native
	./test.x -> test256.txt
	$(CC) $(CFLAGS) -DROWS=512  -DCOLS=32 -DRUNS=100000000 test.c -o test.x -march=native
	./test.x -> test512.txt
	$(CC) $(CFLAGS) -DROWS=1024 -DCOLS=32 -DRUNS=10000000 test.c -o test.x -march=native
	./test.x -> test1024.txt
	$(CC) $(CFLAGS) -DROWS=2048 -DCOLS=32 -DRUNS=10000000 test.c -o test.x -march=native
	./test.x -> test2048.txt

batch_2:
	$(CC) $(CFLAGS) -DCOLS=192 -DROWS=1 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test4.txt
	$(CC) $(CFLAGS) -DCOLS=320 -DROWS=1 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test8.txt
	$(CC) $(CFLAGS) -DCOLS=576 -DROWS=1 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test16.txt
	$(CC) $(CFLAGS) -DCOLS=1088 -DROWS=1 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test32.txt
	$(CC) $(CFLAGS) -DCOLS=2112 -DROWS=1 -DRUNS=1000000000 test.c -o test.x -march=native
	./test.x -> test64.txt
	$(CC) $(CFLAGS) -DCOLS=4160 -DROWS=1 -DRUNS=100000000 test.c -o test.x -march=native
	./test.x -> test128.txt
	$(CC) $(CFLAGS) -DCOLS=8256 -DROWS=1 -DRUNS=100000000 test.c -o test.x -march=native
	./test.x -> test256.txt
	$(CC) $(CFLAGS) -DCOLS=16448 -DROWS=1 -DRUNS=100000000 test.c -o test.x -march=native
	./test.x -> test512.txt
	$(CC) $(CFLAGS) -DCOLS=32768 -DROWS=1 -DRUNS=10000000 test.c -o test.x -march=native
	./test.x -> test1024.txt
	$(CC) $(CFLAGS) -DCOLS=65536 -DROWS=1 -DRUNS=10000000 test.c -o test.x -march=native
	./test.x -> test2048.txt
	