SHELL:=/bin/bash
CFLAGS:=-fPIC -O3 -shared

# add more directives here later
.PHONY : all
all : tracking acquisition

tracking :
	gcc $(CFLAGS) -o tracking.so tracking.c

acquisition :
	gcc $(CFLAGS) -o acquisition.so acquisition.c
