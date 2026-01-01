#ifndef __FILE_HANDLER_HPP__
#define __FILE_HANDLER_HPP__

#include "generals.hpp"
#include <unistd.h>
#include <fcntl.h>
#include <libaio.h>

int io_in_flight_sample = 0;
int io_in_flight_gather = 0;


struct IORequest {
    int block_id;
    float* buffer;
};

class file_handler	{
private:
	std::string datapath;
	std::string filename;
	int fd_sample;
	int fd_gather;
	//for aio
	io_context_t ctx_sample; //AIO context for sample
	io_context_t ctx_gather; //AIO context for gather
	
public:
	std::string filename_graphblock(std::string filename)	{
		std::stringstream ss;
		ss << datapath << filename << ".graphblock";
		return ss.str();
	}

	void set_datapath(std::string _datapath)	{
		datapath = _datapath;
	}

	void open_file(std::string filename)	{
		fd_sample = open(filename_graphblock(filename).c_str(), O_RDONLY | O_DIRECT | O_SYNC);
	}

	void open_file_async(std::string filename) {
		fd_sample = open(filename_graphblock(filename).c_str(), O_RDONLY | O_DIRECT | O_SYNC);
		if (fd_sample < 0) {
			perror("open");
			exit(1);
		}
		// AIO context 초기화
		memset(&ctx_sample, 0, sizeof(ctx_sample));
		if (io_setup(5120, &ctx_sample) < 0) {
			perror("io_setup");
			close(fd_sample);
			exit(1);
    	}
	}

	void open_file_async_gather(std::string filename) {
		fd_gather = open(filename.c_str(), O_RDONLY | O_DIRECT | O_SYNC); //여기
		if (fd_gather < 0) {
			perror("open");
			exit(1);
		}
		// AIO context 초기화
		memset(&ctx_gather, 0, sizeof(ctx_gather));
		if (io_setup(5120, &ctx_gather) < 0) {
			perror("io_setup");
			close(fd_gather);
			exit(1);
    	}
	}

	void close_file()	{
		close(fd_sample);
	}

	void close_file_gather()	{
		io_destroy(ctx_gather);
		close(fd_gather);
	}

	void close_file_async() {
		io_destroy(ctx_sample);
    	close(fd_sample);
	}

	int get_fd()	{
		return fd_sample;
	}

	void read_block(BLOCK * block_ptr, int block_id)	{
		//std::cout << block_ptr <<std::endl;
		if (pread(fd_sample, block_ptr, BLOCKSIZE, (long)((long)block_id * BLOCKSIZE)) <= 0) {
			fprintf(stderr, "reading a block in read_block failed\n");
			std::cout<< fd_sample << std::endl;
			std::cout << block_ptr << std::endl;
			std::cout << BLOCKSIZE << std::endl;
			std::cout << block_id * BLOCKSIZE << std::endl;
			exit(1);
		}
	}

	void read_block_async(BLOCK *block_ptr, int block_id) {
		while (io_in_flight_sample >= 5120) {
			//sched_yield();
			usleep(1000);
		}

		struct iocb* cb = new struct iocb; 
		struct iocb *cbs[1];
		
		io_prep_pread(cb, fd_sample, block_ptr, BLOCKSIZE, (long)((long)block_id * BLOCKSIZE));

		cb->data = (void*)(intptr_t)block_id;
		cbs[0] = cb;

		if (io_submit(ctx_sample, 1, cbs) < 0) {
			perror("io_submit");
			delete cb;
			return;
		}

		io_in_flight_sample++;
	}

	void read_block_async_gather(float* block_ptr, int block_id) {
		while (io_in_flight_gather >= 5120) {
			//sched_yield();
			usleep(1000);
		}

		struct iocb* cb = new struct iocb; 
		struct iocb *cbs[1];
		IORequest* request = new IORequest{block_id, block_ptr};

		io_prep_pread(cb, fd_gather, block_ptr, BLOCKSIZE, (long)((long)block_id * BLOCKSIZE));

		
		cb->data = (void*)request;
		cbs[0] = cb;

		if (io_submit(ctx_gather, 1, cbs) < 0) {
			perror("io_submit");
			delete cb;
			delete request;
			return;
		}

		io_in_flight_gather++;
	}

	int check_io_completion() {
		struct io_event events[1];
		struct timespec timeout;
		timeout.tv_sec = 0; 
		timeout.tv_nsec = 100;
		int block_id = -1;
		int ret = io_getevents(ctx_sample, 1, 1, events, &timeout);
		if (ret > 0) {
			block_id = (int)(intptr_t)events[0].data;
			delete (struct iocb*)events[0].obj;
			io_in_flight_sample--;
		}
		return block_id;
	}

	IORequest check_io_completion_gather() {
		struct io_event events[1];
		struct timespec timeout;
		timeout.tv_sec = 0; 
		timeout.tv_nsec = 100;
		IORequest request;
		request.block_id = -1;
		request.buffer = nullptr;
		int ret = io_getevents(ctx_gather, 1, 1, events, &timeout);
		if (ret > 0) {
			request = *(IORequest*)events[0].data;
			delete (struct iocb*)events[0].obj;
			io_in_flight_gather--;
		}
		return request;
	}

};

#endif