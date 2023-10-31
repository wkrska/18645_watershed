CC		= gcc
CC_FLAGS 	= -mavx -mfma -mavx2 -O3 -std=c99

default: all assemble

# all: clean latency throughput
all: clean latency

latency: latency_and_or.x
	./latency_and_or.x

throughput: throughput_and_or.x
	./throughput_and_or.x

latency_and_or.x:
	$(CC) $(CC_FLAGS) -o latency_and_or.x latency_and_or.c
	
throughput_and_or.x:	
	$(CC) $(CC_FLAGS) -o throughput_and_or.x throughput_and_or.c

assemble:
	objdump -s -d -f --source ./latency_and_or.x > latency_and_or.S
	objdump -s -d -f --source ./throughput_and_or.x > throughput_and_or.S

clean:
	rm -rf *.x *.S
