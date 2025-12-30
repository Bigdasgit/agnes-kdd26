//#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <atomic>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include "file_handler.hpp"

#define	DPFIXED	(sizeof(int) * 4)
#define IN_BLOCKS 1
#define OUT_BLOCKS 2

struct graphmeta	{
	int block_id;					/* block id in the volume */
	vid_t first_nodeid;				/* first nodenumber of the block */
	vid_t last_nodeid;				/* last nodenumber of the block */
};

struct algmeta	{
	std::atomic_flag * lock_ptr;
	int nblocks;
	char * lptemp_ptr;
};

typedef struct metatab {
	int fileindex;					/* the ID of the file in this buffer  */
	int tableindex;
	std::atomic<int> Bfixcnt;	    /* # of active users */
	bool Bdirty;			/* has this block been modified ? */
	bool Brefbit;			/* has buffer been referenced recently  */
	bool Bvalid;			/* dose this buffer contain a valid block  */
} METATAB;

struct graphmeta * graphmeta;	/* index which stores block information about preprocessed graph structure in volume */
struct algmeta ** algmeta_ptr;
struct algmeta * algmeta;		/* index which stores meta-data only used in engine */
BLOCK ** bufferindex;			/* index which stores address information about blocks existing in the buffer */
std::atomic_flag * lock;
char * lptemp;
bool * needed_flag;

//index info
std::string datapath;
std::string filename;
int size_index;					/* size of index which is equal to the number of blocks in preprocessed graph structure */
size_t num_nodes;				/* number of nodes in the graph data */
size_t num_edges;				/* number of edges in the graph data */

int first_index;
int last_index;

int size_index_out;
int size_index_in;

int edge_direction;

size_t size_lptemp;
char * default_lptemp;



int	cpu_clock_hand; 
std::atomic<int> cpu_bf_num_free;	/* number of free buffers */
int cpu_bf_num_bufs;				/* total number of buffers in pool */
BLOCK * cpu_bufferpool;				/* actual buffer pool */
METATAB * cpu_buftable;				/* buffer control table */
file_handler file;	

void set_algmeta(size_t _size_lptemp = 4)	{
		size_lptemp = _size_lptemp;
		default_lptemp = NULL;

		/* make algmeta */
		algmeta_ptr = new struct algmeta*[size_index];
		algmeta = new struct algmeta[size_index];

		/* make others */
		bufferindex = new BLOCK*[size_index];
		lock = new std::atomic_flag[size_index];
		lptemp = new char[size_lptemp * size_index]; 
		needed_flag = new bool[size_index];

		/* initialize algmeta */
		for (int index = 0; index < size_index; index++)	{
			algmeta[index].lock_ptr = NULL;
			algmeta[index].nblocks = 0;
			algmeta[index].lptemp_ptr = NULL;
		}

		/* initialize indices */
		for (int index = 0; index < size_index; index++)	{
			bufferindex[index] = NULL;
		}

		size_t outlargeblocks = 0;

		/* make out-ldi */
		for (int index = 1; index < size_index_out;)	{
			int start_index = index - 1;
			int stop_index = index;

			for (; stop_index < size_index_out; stop_index++)	{
				if (graphmeta[stop_index].first_nodeid != graphmeta[stop_index - 1].first_nodeid)	{
					break;
				}
			}

			for (int i = start_index; i < stop_index; i++)	{
				algmeta[i].lock_ptr = &lock[start_index];
				algmeta[i].lock_ptr->clear();
				algmeta[i].nblocks = stop_index - start_index;
				algmeta[i].lptemp_ptr = &lptemp[size_lptemp * start_index];
				algmeta_ptr[i] = &algmeta[start_index];
			}

			if (stop_index - start_index > 1) outlargeblocks += stop_index - start_index;

			index = stop_index + 1;
		}

		if (size_index_out > 0 && algmeta[size_index_out - 1].nblocks == 0)		{
			algmeta[size_index_out - 1].lock_ptr = &lock[size_index_out - 1];
			algmeta[size_index_out - 1].lock_ptr->clear();
			algmeta[size_index_out - 1].nblocks = 1;
			algmeta[size_index_out - 1].lptemp_ptr = &lptemp[size_lptemp * (size_index_out - 1)];
			algmeta_ptr[size_index_out - 1] = &algmeta[size_index_out - 1];
		}

		size_t inlargeblocks = 0;

		/* make in-ldi */
		for (int index = size_index_out + 1; index < size_index;)	{
			int start_index = index - 1;
			int stop_index = index;

			for (; stop_index < size_index; stop_index++)	{
				if (graphmeta[stop_index].first_nodeid != graphmeta[stop_index - 1].first_nodeid)	{
					break;
				}
			}

			for (int i = start_index; i < stop_index; i++)	{
				algmeta[i].lock_ptr = &lock[start_index];
				algmeta[i].lock_ptr->clear();
				algmeta[i].nblocks = stop_index - start_index;
				algmeta[i].lptemp_ptr = &lptemp[size_lptemp * start_index];
				algmeta_ptr[i] = &algmeta[start_index];
			}

			if (stop_index - start_index > 1) inlargeblocks += stop_index - start_index;

			index = stop_index + 1;
		}

		if (algmeta[size_index - 1].nblocks == 0)		{
			algmeta[size_index - 1].lock_ptr = &lock[size_index - 1];
			algmeta[size_index - 1].lock_ptr->clear();
			algmeta[size_index - 1].nblocks = 1;
			algmeta[size_index - 1].lptemp_ptr = &lptemp[size_lptemp * (size_index - 1)];
			algmeta_ptr[size_index - 1] = &algmeta[size_index - 1];
		}

		/* set each of block intervals: all: first_index <= valid_index < last_index */
        first_index = 0;
		last_index = size_index_out;

		edge_direction = OUT_BLOCKS;
}
void unload_graphmeta()	{
		delete[] algmeta_ptr;
		delete[] algmeta;
		delete[] graphmeta;
		delete[] bufferindex;
		delete[] lock;
		delete[] lptemp;
		delete[] needed_flag;
}
void cpu_setbuffer(size_t NumBuffers)	{
		cpu_clock_hand = -1;
		cpu_bf_num_free = NumBuffers;
		cpu_bf_num_bufs = NumBuffers;

		//cpu_bufferpool = (BLOCK *)malloc((size_t)NumBuffers * sizeof(BLOCK));
        if (posix_memalign((void**)& cpu_bufferpool, 512, (size_t)NumBuffers * sizeof(BLOCK)) != 0) {
            fprintf(stderr, "allocating buffer in cpu_setbuffer failed\n");
            exit(1);
        }
		
		cpu_buftable = (METATAB *)malloc((size_t)NumBuffers * sizeof(METATAB));

		for (size_t index = 0; index < NumBuffers; index++) {
			cpu_buftable[index].Bfixcnt = 0;
			cpu_buftable[index].Bdirty = false;
			cpu_buftable[index].Brefbit = false;
			cpu_buftable[index].Bvalid = false;
		}
}

void cpu_unsetbuffer()	{
		for (int index = 0; index < cpu_bf_num_bufs; index++)	{
			if (cpu_buftable[index].Bvalid && cpu_buftable[index].Bdirty)	{
				if (pwrite(cpu_buftable[index].fileindex, &cpu_bufferpool[index], sizeof(BLOCK), (long)(((long)cpu_buftable[index].tableindex) * BLOCKSIZE)) <= 0) {
					fprintf(stderr, "flushing a block in cpu_unsetbuffer failed\n");
					exit(1);
				}
			}
		}
		free(cpu_bufferpool);
		free(cpu_buftable);
}

void load_graphmeta(std::string _datapath, std::string _filename)	{
		std::string datapath = _datapath;
		std::string filename = _filename;
		//std::cout << datapath << filename << std::endl;
		FILE * fp = NULL;
        std::stringstream graphinfo_filename;
        std::stringstream graphindex_filename;
        graphinfo_filename << datapath << filename << ".graphinfo";
        graphindex_filename << datapath << filename << ".graphindex";
		/* load graph information */

		fp = fopen(graphinfo_filename.str().c_str(), "r+b");
		fseek(fp, 0, SEEK_END);
		if (ftell(fp) == 20) {
			fseek(fp, 0, SEEK_SET);

			if (fread(&size_index_out, sizeof(int), 1, fp) <= 0) {
				fprintf(stderr, "reading size_index_out failed\n");
				exit(1);
			}
			if (fread(&size_index_in, sizeof(int), 1, fp) <= 0) {
				fprintf(stderr, "reading size_index_in failed\n");
				exit(1);
			}
			vid_t _num_nodes;
			if (fread(&_num_nodes, sizeof(vid_t), 1, fp) <= 0) {
				fprintf(stderr, "reading num_nodes failed\n");
				exit(1);
			}
			num_nodes = (size_t)_num_nodes;
			if (fread(&num_edges, sizeof(size_t), 1, fp) <= 0) {
				fprintf(stderr, "reading num_edges failed\n");
				exit(1);
			}

			fseek(fp, 0, SEEK_SET);

			if (fwrite(&size_index_out, sizeof(int), 1, fp) <= 0) {
				fprintf(stderr, "writing size_index_out failed\n");
				exit(1);
			}
			if (fwrite(&size_index_in, sizeof(int), 1, fp) <= 0) {
				fprintf(stderr, "writing size_index_in failed\n");
				exit(1);
			}
			if (fwrite(&num_nodes, sizeof(size_t), 1, fp) <= 0) {
				fprintf(stderr, "writing num_nodes failed\n");
				exit(1);
			}
			if (fwrite(&num_edges, sizeof(size_t), 1, fp) <= 0) {
				fprintf(stderr, "writing num_edges failed\n");
				exit(1);
			}
		}
		else {
			fseek(fp, 0, SEEK_SET);
			if (fread(&size_index_out, sizeof(int), 1, fp) <= 0) {
				fprintf(stderr, "reading size_index_out failed\n");
				exit(1);
			}
			if (fread(&size_index_in, sizeof(int), 1, fp) <= 0) {
				fprintf(stderr, "reading size_index_in failed\n");
				exit(1);
			}
			if (fread(&num_nodes, sizeof(size_t), 1, fp) <= 0) {
				fprintf(stderr, "reading num_nodes failed\n");
				exit(1);
			}
			if (fread(&num_edges, sizeof(size_t), 1, fp) <= 0) {
				fprintf(stderr, "reading num_edges failed\n");
				exit(1);
			}
		}

		fclose(fp);
        size_index = size_index_out + size_index_in;
		//std::cout << size_index_out << std::endl;
		//std::cout << size_index_in << std::endl;
		/* load out-ridtable */
		graphmeta = new struct graphmeta[size_index];
		fp = fopen(graphindex_filename.str().c_str(), "rb");
		if (fread(graphmeta, sizeof(struct graphmeta), size_index, fp) <= 0) {
			fprintf(stderr, "reading graphmeta failed\n");
			exit(1);
        }

		fclose(fp);
}

void create_basic_components(std::string datapath, std::string filename, int buffer_size)	{
		/* load metadata of graph blocks */
		load_graphmeta(datapath, filename);
        set_algmeta();
        size_t memory_buffer_mb = buffer_size; //32768;
		file.set_datapath(datapath);
		file.open_file_async(filename);
		cpu_setbuffer(memory_buffer_mb);
		//std::cout << "create" << std::endl;
	}

void destroy_basic_components()	{
	cpu_unsetbuffer();
    unload_graphmeta();
	file.close_file_async();
	//std::cout << "distroy" << std::endl;
}
std::vector<int> search(vid_t target_node_id)	{
        vid_t start = first_index;
        vid_t end = last_index;

        std::vector<int> block_ids;
        while (start <= end) {
            vid_t mid = start + (end - start) / 2;
            struct graphmeta mid_graphmeta = graphmeta[mid];
            if  (mid_graphmeta.first_nodeid <= target_node_id && target_node_id <= mid_graphmeta.last_nodeid) {
                block_ids.push_back(mid);
            	int left = mid - 1;
            	int right = mid + 1;
				while (left >= first_index && graphmeta[left].first_nodeid <= target_node_id && target_node_id <= graphmeta[left].last_nodeid) {
                	block_ids.push_back(left);
                	left--;
            	}

				while (right <= last_index && graphmeta[right].first_nodeid <= target_node_id && target_node_id <= graphmeta[right].last_nodeid) {
					block_ids.push_back(right);
					right++;
				}
            	break;

            }else if (target_node_id < mid_graphmeta.first_nodeid) {
                end = mid - 1;
            }else {
                start = mid + 1;
            }
        }

		std::sort(block_ids.begin(), block_ids.end());

		return block_ids;

}

int cpu_allocbufs(int fileindex, int _block_id)	{
		int j;	/* loop indices */
		int num_alloc_blocks = 0;
		int clock_start;

		clock_start = cpu_clock_hand == -1 ? 0 : cpu_clock_hand;

		for (j = clock_start + 1;; j++)	{
			if (j == cpu_bf_num_bufs) j = 0;

			if (!cpu_buftable[j].Bvalid)	{
				cpu_buftable[j].Bdirty = false;
				cpu_buftable[j].Bvalid = true;
				cpu_buftable[j].fileindex = fileindex;
				cpu_buftable[j].tableindex = _block_id;
				cpu_buftable[j].Bfixcnt = 1;
				cpu_bf_num_free--;

				bufferindex[_block_id] = &(cpu_bufferpool[j]); // 이 부분
				num_alloc_blocks++;
				/* an unoccupied buffer */
				break;
			}
			else if (cpu_buftable[j].Bfixcnt > 0);
			else if (!cpu_buftable[j].Brefbit) {
				if (cpu_buftable[j].Bdirty)	{ /* a dirty one picked, flush it out */
					if (pwrite(cpu_buftable[j].fileindex, &cpu_bufferpool[j], sizeof(BLOCK), (long)(((long)cpu_buftable[j].tableindex) * BLOCKSIZE)) <= 0) {
						fprintf(stderr, "flushing a block in cpu_allocbufs1 failed\n");
						exit(1);
					}

					cpu_buftable[j].Bdirty = false;
				}
				bufferindex[cpu_buftable[j].tableindex] = NULL;

				cpu_buftable[j].fileindex = fileindex;
				cpu_buftable[j].tableindex = _block_id;
				cpu_buftable[j].Bfixcnt = 1;
				cpu_bf_num_free--;

				bufferindex[_block_id] = &(cpu_bufferpool[j]);
				num_alloc_blocks++;
				break;
				/* a victim found */
			}
			else cpu_buftable[j].Brefbit = false; /* clear ref bit */

			if (j == clock_start) break;
		}
		cpu_clock_hand = j;
		return num_alloc_blocks;
}

void cpu_pinblock(BLOCK * block_ptr) {
	int index = block_ptr - cpu_bufferpool;
	if (cpu_buftable[index].Bfixcnt <= 0) {
		cpu_buftable[index].Bfixcnt = 1;
		cpu_bf_num_free--;
	}
	else cpu_buftable[index].Bfixcnt++;
}

void cpu_unpinblock(BLOCK * block_ptr){
	int index = block_ptr - cpu_bufferpool;
	if (--cpu_buftable[index].Bfixcnt == 0) {
			cpu_bf_num_free++;
			cpu_buftable[index].Brefbit = true;
	}
}


std::tuple<vid_t *, int> find_adj_inblock_LP(vid_t _node_id, BLOCK * block_ptr, struct graphmeta & graphmeta){
	int64_t full_length = block_ptr->slot[0].length;
	vid_t * adjlist_ptr = (vid_t *)&(block_ptr->data[0]);
	return std::make_tuple(adjlist_ptr, full_length);
}

std::tuple<vid_t *, int64_t> find_adj_inblock_SP(vid_t _node_id, BLOCK * block_ptr, struct graphmeta & graphmeta){
	vid_t first_nodeid = graphmeta.first_nodeid;
	int slotno = _node_id - first_nodeid;
	int64_t full_length = block_ptr->slot[-slotno].length;
	vid_t * adjlist_ptr = (vid_t *)&(block_ptr->data[block_ptr->slot[-slotno].offset]);
	return std::make_tuple(adjlist_ptr, full_length);
}