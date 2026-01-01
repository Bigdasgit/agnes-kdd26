#include <queue>
#include <vector>
#include <list>
#include <fstream>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>
#include "buffer.hpp"
#include <mutex>
#include <condition_variable>
#include <Python.h>



float * shared_block_buffer = nullptr;
std::mutex mutexes[1208]; // need to adjust according to max batch num


std::queue<IORequest> completed_requests_gather;
std::mutex completed_mtx_gather;
std::condition_variable completed_cv_gather;
bool running_gather = true;
int need_to_load_blocks = 0;
int total_load_blocks_gather = 0;


float* global_batch_trashcan_buffer = nullptr;

void allocate_global_buffer() {
    size_t buffer_size = 1 * 1024 * 1024 * 1024;  // 1GB

    if (posix_memalign(reinterpret_cast<void**>(&global_batch_trashcan_buffer), 512, buffer_size) != 0) {
        std::cerr << "global buffer allocation failed." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::memset(global_batch_trashcan_buffer, 0, buffer_size);
}

void free_global_buffer() {
    if (global_batch_trashcan_buffer != nullptr) {
        free(global_batch_trashcan_buffer);
        global_batch_trashcan_buffer = nullptr;
        std::cout << "global buffer freed." << std::endl;
    }
}



 
void make_linked_list_for_gather(const std::vector<int64_t>& nodeChunk, size_t i, std::vector<MatrixEntry>& matrix1, std::vector<ListNode>& linkedList, int& nextFreeSlot, std::vector<std::atomic<bool>>& block_indicator, int node_per_block) {
    for (const auto& node : nodeChunk) {
        //total_num_nodes.fetch_add(1);
        
        int blockIndex = node/node_per_block;
        if (!block_indicator[blockIndex].load(std::memory_order_acquire)) {
            block_indicator[blockIndex].store(true, std::memory_order_release);
        }
        int recentIndex = matrix1[blockIndex].recent;
        if (recentIndex == -1) {
            linkedList[nextFreeSlot] = {const_cast<int64_t*>(&nodeChunk[&node - nodeChunk.data()]), -1};
            matrix1[blockIndex].head = nextFreeSlot;
            matrix1[blockIndex].recent = nextFreeSlot;

        } else {
            linkedList[nextFreeSlot] = {const_cast<int64_t*>(&nodeChunk[&node - nodeChunk.data()]), -1};
            linkedList[recentIndex].next = nextFreeSlot;
            matrix1[blockIndex].recent = nextFreeSlot;
            
        }
        nextFreeSlot++;
    }
}

void before_gather_by_thread(std::vector<std::vector<int64_t>>& n_ids, std::vector<std::vector<ListNode>>& linkedLists, int i) {
    size_t n_id_size = n_ids[i].size();
    linkedLists[i].assign(n_id_size, {nullptr, -1}); 
}

void collection(std::vector<std::vector<int64_t>>& n_ids, std::vector<std::atomic<int32_t>>& collected_n_ids, std::atomic<int64_t>& unique_feature){
    tbb::parallel_for(0, int(n_ids.size()), [&](int i) {
        for (int j = 0; j <n_ids[i].size(); j++){
            if (collected_n_ids[n_ids[i][j]] == 0) unique_feature.fetch_add(1); 
            collected_n_ids[n_ids[i][j]].fetch_add(1);
        }
    });

}

int avg_n_ids (std::vector<std::atomic<int32_t>>& collected_n_ids, std::atomic<int64_t>& unique_feature){//합 구하기
    int n_ids_sum = 0;
    for (int i = 0; i < collected_n_ids.size(); i++) {
        n_ids_sum += collected_n_ids[i];
    } 
    //std::cout << "num_gather" << unique_feature << std::endl;
    //std::cout << "num_gather_jung" << n_ids_sum << std::endl;

    return n_ids_sum/unique_feature;
}

void devide(std::vector<std::atomic<int32_t>>& collected_n_ids, int64_t GPU_buffer_size, int64_t CPU_buffer_size, int feature_size, std::atomic<int64_t>& unique_feature,std::vector<int32_t>& feature_GCS, int avg_n_ids, int& GCS_flag){ //먼저 나이브하게 삼등분
    int64_t GPU_feature_num = GPU_buffer_size / feature_size; 
    int64_t CPU_feature_num = CPU_buffer_size / feature_size; 

    if (unique_feature < GPU_feature_num) { 
        for (int i = 0; i<collected_n_ids.size(); i++){ 
            if(collected_n_ids[i]!=0) feature_GCS[i] = 2; 
        }
        
    }
    else { 

        GCS_flag = 1;
        std::atomic<int> GPU_space(GPU_feature_num);
        std::atomic<int> CPU_space(CPU_feature_num);

        for (int i = 0; i<collected_n_ids.size(); i++){ 
            if ((collected_n_ids[i] >= avg_n_ids) && (GPU_space != 0)) { //GPU
                feature_GCS[i] = 2;
                GPU_space.fetch_sub(1);
            }else if((collected_n_ids[i] >= avg_n_ids) && (CPU_space != 0)) { //CPU
                feature_GCS[i] = 1;
                CPU_space.fetch_sub(1);
            }
        }

        for (int i = 0; i<collected_n_ids.size(); i++){ 
            if ((collected_n_ids[i] > 0) && (GPU_space != 0)) { //GPU
                feature_GCS[i] = 2;
                GPU_space.fetch_sub(1);
            }else if((collected_n_ids[i] > 0) && (CPU_space != 0) && (feature_GCS[i] == 0)) { //CPU
                feature_GCS[i] = 1;
                CPU_space.fetch_sub(1);
            }
        }
    }
    if (unique_feature > GPU_feature_num + CPU_feature_num){
        GCS_flag = 0;
    }
}

void load_chunk_from_file(int fd, float* buffer, size_t offset, size_t size) {
    ssize_t bytes_read = pread(fd, buffer + (offset / sizeof(float)), size, offset);
    if (bytes_read == -1) {
        std::cerr << "file read error (" << strerror(errno) << ")" << std::endl;
    }
}


void load_from_file_parallel(int batchnum, int trashcan_count, std::vector<float>& batch_trashcan_queue){
    std::string file_path = "/home/nvme0/AGNES/AGNES_final/trashcan/" +
                            std::to_string(batchnum) + ".dat";


    int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) {
        std::cerr << "file cannot open. " << file_path << " (" << strerror(errno) << ")" << std::endl;
    }


    size_t buffer_size = trashcan_count * BLOCKSIZE;

    size_t num_chunks = trashcan_count;


    tbb::parallel_for(size_t(0), num_chunks, [&](size_t i) {
        size_t offset = i * BLOCKSIZE;
        size_t size = BLOCKSIZE; 
        load_chunk_from_file(fd, global_batch_trashcan_buffer, offset, size);
    });

    close(fd);
}


void load_from_file(int batchnum, int trashcan_count, std::vector<float>& batch_trashcan_queue) {
    std::string file_path = "/home/nvme0/AGNES/AGNES_final/trashcan/" +
                            std::to_string(batchnum) + ".dat";


    int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) {
        std::cerr << "file cannot open. " << file_path << " (" << strerror(errno) << ")" << std::endl;
    }

    size_t buffer_size = trashcan_count * BLOCKSIZE;

    float* batch_trashcan_buffer;
    if (posix_memalign(reinterpret_cast<void**>(&batch_trashcan_buffer), 512, buffer_size) != 0) {
        std::cerr << "memory allocation failed." << std::endl;
        close(fd);
    }


    ssize_t bytes_read = pread(fd, batch_trashcan_buffer, buffer_size, 0);
    if (bytes_read == -1) {
        std::cerr << "file read error (" << strerror(errno) << ")" << std::endl;
        free(batch_trashcan_buffer);
        close(fd);
    }

    std::memcpy(batch_trashcan_queue.data(), batch_trashcan_buffer, buffer_size);
    
    free(batch_trashcan_buffer);

    close(fd);
}


void flush_can(std::vector<float>& batch_trashcan, int batchnum, int trashcan_count) {
    std::string file_path = "/home/nvme0/AGNES/AGNES_final/trashcan/" +
                            std::to_string(batchnum) + ".dat";


    int fd = open(file_path.c_str(), O_WRONLY | O_CREAT, 0666);
    if (fd == -1) {
        std::cerr << "file cannot open. " << file_path << " (" << strerror(errno) << ")" << std::endl;
        return;
    }

    off_t offset = static_cast<off_t>(trashcan_count * batch_trashcan.size() * sizeof(float));


    ssize_t bytes_written = pwrite(fd, batch_trashcan.data(), batch_trashcan.size() * sizeof(float), offset);
    if (bytes_written != BLOCKSIZE) {
        std::cerr << "file write error (" << strerror(errno) << ")" << std::endl;
        close(fd);
        return;
    }

    close(fd);
}



void load_and_gather(std::vector<std::vector<MatrixEntry>>& matrix1, 
                    std::vector<std::vector<ListNode>>& linkedLists, 
                    std::vector<std::vector<int64_t>>& n_ids,
                    std::vector<int32_t>& feature_GCS, 
                    std::vector<float>& GPU_feature_q, 
                    std::vector<float>& CPU_feature_q, 
                    std::vector<std::vector<float>>& batch_trashcan,
                    std::atomic<int64_t>& GPU_feature_q_indicator,
                    std::atomic<int64_t>& CPU_feature_q_indicator,
                    std::vector<std::atomic<int64_t>>& batch_trashcan_indicator,  
                    std::vector<int64_t>& feature_q_index,
                    std::vector<std::vector<int64_t>>& storage_q_index,
                    std::vector<std::atomic<int>>& trashcan_count,
                    std::vector<std::atomic<bool>>& flushing,
                    std::vector<std::atomic<int>>& write_count, 
                    int block_id, int feature_dim, float* feature_block_buffer){
    
    int node_per_block = BLOCKSIZE/(feature_dim*sizeof(float));                   
    std::vector<bool> used_flag(node_per_block, 0); 
 
    int64_t node_id = -1;
    int64_t batch_location = -1;
    int32_t next_index = -1;
    int32_t node_index = -1;
    int64_t GPU_ind = 0;
    int64_t CPU_ind = 0;
    int64_t ST_ind = 0;
    int64_t Act_ST_ind = 0;
    int local_trashcan_count = 0;
    
    for (int i = 0 ; i < matrix1.size() ; i++){
        if (matrix1[i][block_id].head != -1){
            node_id = *(linkedLists[i][matrix1[i][block_id].head].nodeAddress);
            batch_location = (linkedLists[i][matrix1[i][block_id].head].nodeAddress - n_ids[i].data());
            next_index = linkedLists[i][matrix1[i][block_id].head].next;
            node_index = node_id-block_id*node_per_block;
            if (used_flag[node_index]==0){
                if (feature_GCS[node_id]==2){ //GPU 
                    GPU_ind = GPU_feature_q_indicator.fetch_add(feature_dim);
                    feature_q_index[node_id] = GPU_ind;
                    std::memcpy(GPU_feature_q.data() + GPU_ind, feature_block_buffer + node_index * feature_dim, feature_dim * sizeof(float));
                    used_flag[node_index] = 1;
                }else if (feature_GCS[node_id]==1){ //CPU
                    CPU_ind = CPU_feature_q_indicator.fetch_add(feature_dim);
                    feature_q_index[node_id] = CPU_ind;
                    std::memcpy(CPU_feature_q.data() + CPU_ind, feature_block_buffer + node_index * feature_dim, feature_dim * sizeof(float));
                    used_flag[node_index] = 1;
                } 
                else if (feature_GCS[node_id] == 0) {
                    ST_ind = batch_trashcan_indicator[i].fetch_add(feature_dim);
                    storage_q_index[i][batch_location] = ST_ind; 
                    while(1){
                        local_trashcan_count = trashcan_count[i].load(); 
                        if (ST_ind >= local_trashcan_count * (BLOCKSIZE / sizeof(float)) && ST_ind < (local_trashcan_count+1)* (BLOCKSIZE / sizeof(float))){ 
                            mutexes[i].lock();
                            std::memcpy(batch_trashcan[i].data() + ST_ind - local_trashcan_count * (BLOCKSIZE / sizeof(float)), feature_block_buffer + node_index * feature_dim, feature_dim * sizeof(float));  
                            int current_count = write_count[i].fetch_add(1) + 1; 
                            if (current_count > node_per_block) std::cout << current_count << std::endl;
                            if (current_count==node_per_block){ //
                                    flush_can(batch_trashcan[i], i, trashcan_count[i].fetch_add(1));
                                    write_count[i].store(0);
                                }
                            mutexes[i].unlock();
                            break;
                            
                        }
                    }
                }
            }
            
            while(next_index != -1){
                node_id = *(linkedLists[i][next_index].nodeAddress);
                batch_location = (linkedLists[i][next_index].nodeAddress - n_ids[i].data());
                next_index = linkedLists[i][next_index].next;
                node_index = node_id-block_id*node_per_block;
                if (used_flag[node_index]==0){   
                    if (feature_GCS[node_id]==2){ //GPU
                        GPU_ind = GPU_feature_q_indicator.fetch_add(feature_dim);
                        feature_q_index[node_id] = GPU_ind;
                        std::memcpy(GPU_feature_q.data() + GPU_ind, feature_block_buffer + node_index * feature_dim, feature_dim * sizeof(float));
                        used_flag[node_index] = 1;
                    }else if (feature_GCS[node_id]==1){ //CPU
                        CPU_ind = CPU_feature_q_indicator.fetch_add(feature_dim);
                        feature_q_index[node_id] = CPU_ind;
                        std::memcpy(CPU_feature_q.data() + CPU_ind, feature_block_buffer + node_index * feature_dim, feature_dim * sizeof(float));
                        used_flag[node_index] = 1;
                    }
                    else if (feature_GCS[node_id] == 0) {
                        ST_ind = batch_trashcan_indicator[i].fetch_add(feature_dim); 
                        storage_q_index[i][batch_location] = ST_ind; 
                        while(1){
                            local_trashcan_count = trashcan_count[i].load();
                            if (ST_ind >= local_trashcan_count * (BLOCKSIZE / sizeof(float)) && ST_ind < (local_trashcan_count+1)* (BLOCKSIZE / sizeof(float))){ 
                                mutexes[i].lock();
                                std::memcpy(batch_trashcan[i].data() + ST_ind - local_trashcan_count * (BLOCKSIZE / sizeof(float)), feature_block_buffer + node_index * feature_dim, feature_dim * sizeof(float)); 
                                int current_count = write_count[i].fetch_add(1) + 1; 
                                if (current_count > node_per_block) std::cout << current_count << std::endl;
                                if (current_count==node_per_block){
                                    flush_can(batch_trashcan[i], i, trashcan_count[i].fetch_add(1));
                                    write_count[i].store(0);
                                }
                                mutexes[i].unlock();
                                break;
                                
                            }
                        }
                    }
                }
            }
        }
    }
    free(feature_block_buffer);
}


void wait_for_comp_gather() {
    int num_finished_tasks = 0; 
    while (running_gather) {
        auto req = file.check_io_completion_gather();
        if (req.block_id >= 0){
            std::unique_lock<std::mutex> lock(completed_mtx_gather);
            completed_requests_gather.push(req); 
            completed_cv_gather.notify_one(); 
            num_finished_tasks++;
        }
        if (num_finished_tasks == need_to_load_blocks) {
            running_gather = false;
            completed_cv_gather.notify_all(); 
        }

    }
}

void io_submission_thread_gather(std::vector<std::atomic<bool>>& block_indicator) {
    for (int i = 0; i < block_indicator.size(); i++) {
        if (block_indicator[i]) {
            float* feature_block_buffer;
            if (posix_memalign((void**)&feature_block_buffer, BLOCKSIZE, BLOCKSIZE+1) != 0) { 
                fprintf(stderr, "allocating buffer in cpu_setbuffer failed\n");
                exit(1);
            }
            file.read_block_async_gather(feature_block_buffer, i);
        }
    }

}


void work_by_thread_gather(std::vector<std::vector<MatrixEntry>>& matrix1, 
                    std::vector<std::vector<ListNode>>& linkedLists,
                    std::vector<std::vector<int64_t>>& n_ids, 
                    std::vector<std::atomic<bool>>& block_indicator, 
                    std::vector<int32_t>& feature_GCS, 
                    std::vector<float>& GPU_feature_q, 
                    std::vector<float>& CPU_feature_q, 
                    std::vector<std::vector<float>>& batch_trashcan, 
                    std::atomic<int64_t>& GPU_feature_q_indicator,
                    std::atomic<int64_t>& CPU_feature_q_indicator,
                    std::vector<std::atomic<int64_t>>& batch_trashcan_indicator, 
                    std::vector<int64_t>& feature_q_index,
                    std::vector<std::vector<int64_t>>& storage_q_index, 
                    std::vector<std::atomic<int>>& trashcan_count, 
                    std::vector<std::atomic<bool>>& flushing,
                    std::vector<std::atomic<int>>& write_count,
                    int feature_dim) {

    while (running_gather || !completed_requests_gather.empty()) {
        int block_id = -1;
        float* feature_block_buffer = nullptr;
        {
            std::unique_lock<std::mutex> lock(completed_mtx_gather);
            completed_cv_gather.wait(lock, [] { return !completed_requests_gather.empty() || !running_gather; });

            if (!completed_requests_gather.empty()) {
                block_id = completed_requests_gather.front().block_id;
                feature_block_buffer = completed_requests_gather.front().buffer;
                completed_requests_gather.pop();
            } else {
                continue; 
            }
        }


        load_and_gather(matrix1,
                        linkedLists,
                        n_ids,
                        feature_GCS,
                        GPU_feature_q,
                        CPU_feature_q,
                        batch_trashcan,
                        GPU_feature_q_indicator,
                        CPU_feature_q_indicator,
                        batch_trashcan_indicator,
                        feature_q_index,
                        storage_q_index, 
                        trashcan_count,
                        flushing,
                        write_count, 
                        block_id,  
                        feature_dim,
                        feature_block_buffer);
    }
}


void gather_by_block_async(std::vector<std::vector<MatrixEntry>>& matrix1, 
                    std::vector<std::vector<ListNode>>& linkedLists,
                    std::vector<std::vector<int64_t>>& n_ids, 
                    std::vector<std::atomic<bool>>& block_indicator, 
                    std::vector<int32_t>& feature_GCS, 
                    std::vector<float>& GPU_feature_q, 
                    std::vector<float>& CPU_feature_q, 
                    std::vector<std::vector<float>>& batch_trashcan, 
                    std::atomic<int64_t>& GPU_feature_q_indicator,
                    std::atomic<int64_t>& CPU_feature_q_indicator,
                    std::vector<std::atomic<int64_t>>& batch_trashcan_indicator, 
                    std::vector<int64_t>& feature_q_index,
                    std::vector<std::vector<int64_t>>& storage_q_index, 
                    std::vector<std::atomic<int>>& trashcan_count, 
                    int feature_dim, std::string feature_path, int num_workers){

    running_gather = true;
    need_to_load_blocks = 0;

    for (int i = 0; i < block_indicator.size(); i++){
        if (block_indicator[i] == true){
            need_to_load_blocks++;
        }
    }
    total_load_blocks_gather+=need_to_load_blocks;

    std::string file_path = feature_path + "features" + std::to_string(feature_dim) + ".dat";
    file.open_file_async_gather(file_path); 

    std::vector<std::atomic<bool>> flushing(matrix1.size());
    std::vector<std::atomic<int>> write_count(matrix1.size());
    for (int i = 0; i < matrix1.size(); i++) {
        trashcan_count[i].store(0);
        write_count[i].store(0);
        flushing[i].store(false);
    }
    std::vector<std::thread> workers_gather;
    for (int i = 0; i < num_workers; ++i) {
        workers_gather.emplace_back([&] { work_by_thread_gather(matrix1, 
                    linkedLists,
                    n_ids, 
                    block_indicator, 
                    feature_GCS, 
                    GPU_feature_q, 
                    CPU_feature_q, 
                    batch_trashcan, 
                    GPU_feature_q_indicator,
                    CPU_feature_q_indicator,
                    batch_trashcan_indicator, 
                    feature_q_index,
                    storage_q_index, 
                    trashcan_count, 
                    flushing,
                    write_count,
                    feature_dim); });
    }
    std::thread submission_thread_gather (io_submission_thread_gather, std::ref(block_indicator)); 
    wait_for_comp_gather();
    
    submission_thread_gather.join();
    for (auto& worker : workers_gather) {
        worker.join();
    }
    file.close_file_gather();
    
}


void gather_by_batch(std::vector<int64_t>& Batch, 
                    std::vector<float>& GPU_feature_q, 
                    std::vector<float>& CPU_feature_q, 
                    std::vector<int64_t>& feature_q_index,
                    std::vector<int64_t>& storage_q_index, 
                    std::vector<std::vector<float>>& batch_trashcan,
                    std::vector<int32_t>& feature_GCS,
                    std::vector<float>& batch_feature_queue,
                    std::vector<float>& batch_trashcan_queue,
                    int batchnum, int feature_dim, int GCS_flag, int num_trashcan, int num_workers){
    
    PyThreadState* _save;
    _save = PyEval_SaveThread(); // GIL 
 
    int feature_size_bytes = feature_dim * sizeof(float);
    if (GCS_flag == 0){ 
        load_from_file_parallel(batchnum, num_trashcan, batch_trashcan_queue);
        std::memcpy(global_batch_trashcan_buffer + num_trashcan * (BLOCKSIZE / sizeof(float)), batch_trashcan[batchnum].data(), BLOCKSIZE);
    }
    int64_t N = Batch.size();
    tbb::task_arena limited_arena(num_workers);
    limited_arena.execute([&](){
        tbb::parallel_for(int64_t(0), N, [&](int64_t i) {
            if (feature_GCS[Batch[i]]==2){
            }else if (feature_GCS[Batch[i]]==1){
                std::memcpy(batch_feature_queue.data() + i*feature_dim, CPU_feature_q.data() + feature_q_index[Batch[i]], feature_size_bytes);
            }else {
                std::memcpy(batch_feature_queue.data() + i * feature_dim, global_batch_trashcan_buffer + storage_q_index[i], feature_size_bytes);

            }
        });
    });
    PyEval_RestoreThread(_save); // GIL 
    
}

void after_gather_by_thread(std::vector<std::vector<ListNode>>& linkedLists, std::vector<std::vector<MatrixEntry>>& matrix1, std::vector<std::vector<float>>& batch_trashcan, int i) {
    linkedLists[i].clear();
    matrix1[i].clear();
}

void after_gather(int batchNum, std::vector<std::vector<ListNode>>& linkedLists, std::vector<std::vector<MatrixEntry>>& matrix1, std::vector<float>& CPU_feature_q, std::vector<std::vector<float>>& batch_trashcan) {
    tbb::parallel_for(0, batchNum, [&](int i) {
        after_gather_by_thread(linkedLists, matrix1, batch_trashcan,i);
    });
}

void copy_atomic_to_non_atomic(const std::vector<std::atomic<int>>& src, std::vector<int>& dest) {
    dest.resize(src.size());
    
    for (size_t i = 0; i < src.size(); ++i) {
        dest[i] = src[i].load(); 
    }
}






int gather(std::vector<std::vector<int64_t>>& n_ids, int maxValue,
           std::vector<float>& GPU_feature_q,
           std::vector<float>& CPU_feature_q,
           std::vector<int64_t>& feature_q_index,  
           std::vector<std::vector<int64_t>>& storage_q_index,
           std::vector<std::vector<float>>& batch_trashcan,
           std::vector<int>& trashcan_count_g, 
           std::vector<int32_t>& feature_GCS,
           int feature_dim,
           int feature_buffer_size,
           int num_workers,
           std::string feature_path) {
    
    PyThreadState* _save;
    _save = PyEval_SaveThread(); // GIL 
    
    auto start = std::chrono::high_resolution_clock::now();
    int batchNum = n_ids.size();
    int64_t GPU_buffer_size = 0LL * 1024 * 1024 * 1024; 
    int64_t CPU_buffer_size = static_cast<int64_t>(feature_buffer_size) * 1024 * 1024; 
    int feature_size = feature_dim * sizeof(float); 
    std::vector<std::atomic<int>> trashcan_count(batchNum); 

    int GCS_flag = 2; 
    int node_per_block = BLOCKSIZE/feature_size;
    int totalBlock = maxValue/node_per_block+1; 
    std::atomic<int64_t> GPU_feature_q_indicator(0);
    std::atomic<int64_t> CPU_feature_q_indicator(0);
    std::vector<std::atomic<int64_t>> batch_trashcan_indicator(batchNum); 

    tbb::parallel_for(0, batchNum, [&](int i) {
        batch_trashcan_indicator[i].store(0);
        trashcan_count[i].store(0);
    });
    std::vector<std::atomic<bool>> block_indicator(totalBlock); 


    /*for collection*/
    std::atomic<int64_t> unique_feature(0); 
    std::vector<std::atomic<int32_t>> collected_n_ids(maxValue); 

    tbb::parallel_for(0, maxValue, [&](int i) {
        collected_n_ids[i].store(0); 
    });
    collection(n_ids,collected_n_ids, unique_feature); 
    int unique_avg = avg_n_ids(collected_n_ids, unique_feature);
    devide(collected_n_ids, GPU_buffer_size, CPU_buffer_size, feature_size, unique_feature, feature_GCS, unique_avg, GCS_flag);

    if (GCS_flag == 2) {
        GPU_feature_q.resize(GPU_buffer_size);
    }else if (GCS_flag == 1) {
        //std::cout << "GCS1" << std::endl;
        GPU_feature_q.reserve(GPU_buffer_size/sizeof(float)); 
        CPU_feature_q.reserve(CPU_buffer_size/sizeof(float)); 
        GPU_feature_q.resize(GPU_buffer_size/sizeof(float)); 
        CPU_feature_q.resize(CPU_buffer_size/sizeof(float)); 
    } else {
        //std::cout << "GCS0" << std::endl;
        GPU_feature_q.reserve(GPU_buffer_size/sizeof(float)); 
        CPU_feature_q.reserve(CPU_buffer_size/sizeof(float)); 
        GPU_feature_q.resize(GPU_buffer_size/sizeof(float)); 
        CPU_feature_q.resize(CPU_buffer_size/sizeof(float)); 
        for (int i = 0; i < batchNum; i++){
            batch_trashcan[i].resize(BLOCKSIZE/sizeof(float), -1); 
            storage_q_index[i].resize(n_ids[i].size()); 
        }
    }
 
    std::vector<std::vector<MatrixEntry>> matrix_for_head(batchNum, std::vector<MatrixEntry>(totalBlock, {-1, -1}));
    std::vector<std::vector<ListNode>> linkedLists(batchNum, std::vector<ListNode>(n_ids.size(), {nullptr, -1})); 
    std::vector<int32_t> nextFreeSlots(batchNum, 0);

    tbb::task_arena limited_arena(16);
    limited_arena.execute([&](){
        tbb::parallel_for(0, batchNum, [&](int i) {
            before_gather_by_thread(n_ids, linkedLists, i);
        });
    });

    limited_arena.execute([&](){
        tbb::parallel_for(0, batchNum, [&](int i) {
            make_linked_list_for_gather(n_ids[i], i, matrix_for_head[i], linkedLists[i], nextFreeSlots[i], block_indicator, node_per_block);
        });
    });

    gather_by_block_async(matrix_for_head, 
                    linkedLists, 
                    n_ids,
                    block_indicator, 
                    feature_GCS, 
                    GPU_feature_q, 
                    CPU_feature_q, 
                    batch_trashcan, 
                    GPU_feature_q_indicator, 
                    CPU_feature_q_indicator, 
                    batch_trashcan_indicator,  
                    feature_q_index,
                    storage_q_index,
                    trashcan_count,
                    feature_dim,
                    feature_path,
                    num_workers);


    copy_atomic_to_non_atomic(trashcan_count, trashcan_count_g);
    after_gather(batchNum, linkedLists, matrix_for_head, CPU_feature_q, batch_trashcan);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();

    PyEval_RestoreThread(_save); // GIL
    //std::cout << "num_blocks_gather = " << total_load_blocks_gather << std::endl;
    malloc_trim(0);
    return GCS_flag;

}
