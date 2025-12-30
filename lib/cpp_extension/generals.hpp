#ifndef __GENERALS_HPP__
#define __GENERALS_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <atomic>
#include <map>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include <climits>
#include <thread>
#include <omp.h>
#include <malloc.h>
#include <fstream>
#include <string>

#define WORDTYPE unsigned long long int
#define WORDRANGE_LOG2 6
#define WORDFILTER 63
#define WORDBIT(node_id) ((WORDTYPE)(1) << ((node_id) & WORDFILTER))

#define vid_t uint64_t
#define tokenvid(str,endptr,base) strtoul(str,endptr,base)

#ifdef val_32
#define val_t float
#define tokenval(str,endptr) strtof(str,endptr)
#else
#define val_t double
#define tokenval(str,endptr) strtod(str,endptr)
#endif

#define BLOCKSIZE 1048576  //4194304 //524288 //1048576 // //65536//	1048576  524288 262144 131072 65536 32768 16384 2097152 4194304
#define	DPFIXED	(sizeof(int) * 4)

#define BUFFERSIZE 5000
#define MILLION 524288 //524288 262144 131072 65536 32768 1048576 2097152

struct MatrixEntry {
    int head;
    int recent;
    //int length;
};

struct vectorEntry {
    int batch;
    int index;
    //int length;
};

struct ListNode {
    int64_t* nodeAddress;
    int next;
};



typedef struct {
	int length;
	int offset;
} SLOT;

typedef struct {
	char	data[BLOCKSIZE - DPFIXED];
	SLOT	slot[1];					/* slot 0 -- index backwards */
	int	free;			/* pointer to data free area */
	int ridcnt;
} BLOCK;

typedef struct edge {
	vid_t src;
	vid_t dst;

	edge() {
		src = 0;
		dst = 0;
	}

	edge(vid_t _src, vid_t _dst) {
		src = _src;
		dst = _dst;
	}
}edge;

bool less_edge_with_novalue(vid_t i, vid_t j)	{
	return i < j;
}

typedef struct edge_with_value {
	vid_t src;
	vid_t dst;
	val_t val;

	edge_with_value() {
		src = 0;
		dst = 0;
		val = 0;
	}

	edge_with_value(vid_t _src, vid_t _dst, val_t _val) {
		src = _src;
		dst = _dst;
		val = _val;
	}
}edge_with_value;

typedef struct adj_with_value{
	vid_t vid;
	val_t val;

	adj_with_value(vid_t _vid, val_t _val)	{
		vid = _vid;
		val = _val;
	}
}adj_with_value;

bool less_edge_with_value(adj_with_value i, adj_with_value j)	{
	return i.vid < j.vid;
}

void print_memory_usage() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.find("VmRSS:") == 0) { // Resident Set Size (RSS)
            std::cout << "Current memory usage (RSS): " << line << std::endl;
        } else if (line.find("VmSize:") == 0) { // Virtual Memory Size (VmSize)
            std::cout << "Virtual memory size (VmSize): " << line << std::endl;
        }
    }
    status_file.close();
}




#ifdef weighted

#define adj_t adj_with_value
#define edge_t edge_with_value
#define less_edge less_edge_with_value

#else

#define adj_t vid_t
#define edge_t edge
#define less_edge less_edge_with_novalue

#endif

#endif