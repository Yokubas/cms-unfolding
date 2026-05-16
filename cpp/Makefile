CC=g++

CFLAGS=-c -g -Wall `root-config --cflags`

LDFLAGS=`root-config --glibs`

all: read_root 

read_root: read_root.cpp
	$(CC) $(CFLAGS) -o read_root.o read_root.cpp
	$(CC) -o read_root read_root.o $(LDFLAGS)

clean:
	rm -f ./*~ ./*.o ./read_root
	