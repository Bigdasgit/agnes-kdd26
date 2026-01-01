#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <queue>
#include <boost/asio.hpp>
#include <chrono>
#include <unistd.h>
#include <unordered_set>
#include <unordered_map>
#include <atomic>
#include <immintrin.h>
#include <mutex>
#include <torch/extension.h>
#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>
#include "block_based_gather.cpp"

tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);



std::atomic<int64_t> total_load_time_ns(0);
std::atomic<int64_t> total_sample_time_ns(0);
std::atomic<int64_t> total_pushback_time_ns(0);
std::atomic<int64_t> total_wo_pushback_time_ns(0);
std::atomic<int64_t> last_time_ns(0);
double total_cols_time = 0;
std::atomic<int> total_num_nodes(0);
int total_num_blocks = 0;
std::atomic<int64_t> num_total_act_node(0);
std::vector<std::mutex> out_ptr_mutex(1208); 


//for AsyncIO
std::queue<int> completed_requests;
std::mutex completed_mtx;
std::condition_variable completed_cv;
bool running = true;

//for gathering
std::vector<float> GPU_feature_q;            // feature queue creation
std::vector<float> CPU_feature_q;            // feature queue creation
std::vector<int64_t> feature_q_index;            // feature queue index creation
std::vector<std::vector<int64_t>> storage_q_index;
std::vector<std::vector<float>> batch_trashcan;      // feature queue creation
std::vector<int> trashcan_count; 
std::vector<int32_t> feature_GCS; 
std::vector<float> batch_feature_queue1;
std::vector<float> batch_feature_queue2;
std::vector<float> batch_trashcan_queue;


int GCS_flag = 2;



void make_linked_list(const std::vector<int64_t>& nodeChunk, size_t i, std::vector<MatrixEntry>& matrix1, std::vector<ListNode>& linkedList, int& nextFreeSlot, std::vector<std::atomic<bool>>& block_indicator) {

    for (const auto& node : nodeChunk) {
        std::vector<int> blocks = search(node);
        int blockIndex = blocks[0]; 
        if (!block_indicator[blockIndex].load(std::memory_order_acquire)) {
            block_indicator[blockIndex].store(true, std::memory_order_release);
        }
        int recentIndex = matrix1[blockIndex].recent;
        if (recentIndex == -1) {
            linkedList[nextFreeSlot] = {const_cast<int64_t*>(&nodeChunk[&node - nodeChunk.data()]), -1};//나중에 검토
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

void load_and_sample(std::vector<std::vector<MatrixEntry>>& matrix1, 
                     std::vector<std::vector<ListNode>>& linkedLists, 
                     int block_id, 
                     std::vector<std::vector<std::vector<int64_t>>>& cols, 
                     std::vector<std::atomic<int>>& n_ids_index, 
                     std::vector<std::vector<int64_t>>& n_ids,
                     std::vector<std::vector<int64_t>>& Batches,  
                     std::vector<tbb::concurrent_hash_map<int64_t, int>>& n_id_map, 
                     bool readflag, int sample_size){
    if(readflag == true) file.read_block_async(bufferindex[block_id], block_id);
    
    bool LPflag = false;
    int LP_len = 0;
    if (algmeta[block_id].nblocks != 1) {
        LPflag = true;
        LP_len = algmeta[block_id].nblocks;
    }
    int64_t node_id = -1;
    int64_t batch_location = -1;
    int next_index = -1;
    vid_t * node_adjlist_ptr;
    int node_full_length;
    int64_t current_n_id_index;
    int64_t c;
    std::mt19937 gen(block_id);


    for (int i = 0 ; i < matrix1.size() ; i++){
        if (matrix1[i][block_id].head != -1){
            node_id = *(linkedLists[i][matrix1[i][block_id].head].nodeAddress);
            next_index = linkedLists[i][matrix1[i][block_id].head].next;
            batch_location = (linkedLists[i][matrix1[i][block_id].head].nodeAddress - Batches[i].data());

            std::tie(node_adjlist_ptr, node_full_length) = find_adj_inblock_SP(node_id, bufferindex[block_id], graphmeta[block_id]);
            std::uniform_int_distribution<int64_t> dis(0, node_full_length -1);
            //full sample
            std::unordered_set<int64_t> perm; 
            __mmask8 mask1, mask2;
            if (node_full_length <= sample_size) {
                for (int64_t j = 0; j < node_full_length; j++)
                perm.insert(j);
            }    
            
            else if (LPflag) {
                node_full_length = BLOCKSIZE/sizeof(int64_t)-2000; //임시방편
                for (int64_t j = node_full_length - sample_size; j < node_full_length; j++) {
                    
                    int64_t rand_val;
                    do {
                        rand_val = dis(gen);
                    } while (!perm.insert(rand_val).second);
                } 
            }
            else {
                for (int64_t j = node_full_length - sample_size; j < node_full_length; j++) {
                    
                    int64_t rand_val;
                    do {
                        rand_val = dis(gen);
                    } while (!perm.insert(rand_val).second);
                    
                }   
            }
            for (const int64_t& p : perm) {
                c = node_adjlist_ptr[p]; 

                tbb::concurrent_hash_map<int64_t, int>::accessor accessor;
                if (n_id_map[i].insert(accessor, c)) {
                    current_n_id_index = n_ids_index[i].fetch_add(1);
                    accessor->second = current_n_id_index;
                    n_ids[i][current_n_id_index] = c;
                } else {
                    current_n_id_index = accessor->second;
                }
                cols[i][batch_location].push_back(current_n_id_index); 
            }
            
            
            

            while(next_index != -1){
                node_id = *(linkedLists[i][next_index].nodeAddress);
                batch_location = (linkedLists[i][next_index].nodeAddress - Batches[i].data());
                next_index = linkedLists[i][next_index].next;

                std::tie(node_adjlist_ptr, node_full_length) = find_adj_inblock_SP(node_id, bufferindex[block_id], graphmeta[block_id]);
                std::uniform_int_distribution<int64_t> dis(0, node_full_length -1);
                std::unordered_set<int64_t> perm; 
                __mmask8 mask1, mask2;
                
                if (node_full_length <= sample_size) {
                    for (int64_t j = 0; j < node_full_length; j++)
                    perm.insert(j);
                }   
                else if (LPflag) {
                    node_full_length = BLOCKSIZE/sizeof(int64_t)-2000; //임시방편
                    for (int64_t j = node_full_length - sample_size; j < node_full_length; j++) {
                        int64_t rand_val;
                        do {
                            rand_val = dis(gen);
                        } while (!perm.insert(rand_val).second);
                    } 
                }
                else {
                    for (int64_t j = node_full_length - sample_size; j < node_full_length; j++) {    
                        int64_t rand_val;
                        do {
                            rand_val = dis(gen);
                        } while (!perm.insert(rand_val).second); 
                        
                    }
                }
                for (const int64_t& p : perm) {
                    c = node_adjlist_ptr[p]; 

                    tbb::concurrent_hash_map<int64_t, int>::accessor accessor;
                    if (n_id_map[i].insert(accessor, c)) {
                        current_n_id_index = n_ids_index[i].fetch_add(1);
                        accessor->second = current_n_id_index;
                        n_ids[i][current_n_id_index] = c;
                    } else {
                        current_n_id_index = accessor->second;
                    }
                    cols[i][batch_location].push_back(current_n_id_index); 
                }
                
            }
        }
    }
    cpu_unpinblock(bufferindex[block_id]);
    

}



void io_submission_thread(const std::vector<int>& block_ids) {
    for (int block_id : block_ids) {
        int alloc = 0;
        while (alloc != 1) {
            alloc = cpu_allocbufs(file.get_fd(), block_id);
        }
        file.read_block_async(bufferindex[block_id], block_id);
    }
}

void work_by_thread(std::vector<std::vector<MatrixEntry>>& matrix1, 
                      std::vector<std::vector<ListNode>>& linkedLists, 
                      std::vector<std::vector<std::vector<int64_t>>>& cols, 
                      std::vector<std::atomic<int>>& n_ids_index, 
                      std::vector<std::vector<int64_t>>& n_ids,
                      std::vector<std::vector<int64_t>>& Batches, 
                      std::vector<tbb::concurrent_hash_map<int64_t, int>>& n_id_map,
                      int sample_size) {

    while (running || !completed_requests.empty()) {
        int block_id = -1;
        {
            std::unique_lock<std::mutex> lock(completed_mtx);
            completed_cv.wait(lock, [] { return !completed_requests.empty() || !running; });

            if (!completed_requests.empty()) {
                block_id = completed_requests.front();
                completed_requests.pop();
            } else {
                continue; 
            }
        }


        load_and_sample(matrix1, linkedLists, block_id, cols, n_ids_index, n_ids, Batches, n_id_map, false, sample_size);
    }
}

void wait_for_comp(int vec_size) {
    int num_finished_tasks = 0; 
    while (running) {
        int block_id = file.check_io_completion();
        if (block_id >= 0){
            std::unique_lock<std::mutex> lock(completed_mtx);
            completed_requests.push(block_id); 
            completed_cv.notify_one(); 
            num_finished_tasks++;
        }
        if (num_finished_tasks == vec_size) {
            running = false;
            completed_cv.notify_all(); 
        }

    }
}

void process_by_block_async(std::vector<std::vector<MatrixEntry>>& matrix1, 
                      std::vector<std::vector<ListNode>>& linkedLists, 
                      std::vector<std::atomic<bool>>& block_indicator, 
                      std::vector<std::vector<std::vector<int64_t>>>& cols, 
                      std::vector<std::atomic<int>>& n_ids_index, 
                      std::vector<std::vector<int64_t>>& n_ids,
                      std::vector<std::vector<int64_t>>& Batches, 
                      std::vector<tbb::concurrent_hash_map<int64_t, int>>& n_id_map,
                      int num_workers, int sample_size) {
    running = true;
    std::vector<int> null_blocks;
    std::vector<int> non_null_blocks;

    for (int i = 0; i < block_indicator.size(); i++){
        if (block_indicator[i] == true){
            if (bufferindex[i] == NULL){
                null_blocks.push_back(i);
            } else {
                non_null_blocks.push_back(i);
            }
        }
    }
    total_num_blocks+=null_blocks.size();
    //std::cout << null_blocks.size() << std::endl;
    tbb::task_arena limited_arena(num_workers);
    limited_arena.execute([&](){
        tbb::parallel_for(size_t(0), non_null_blocks.size(), [&](int j) {
            int i = non_null_blocks[j];
            load_and_sample(matrix1, linkedLists, i, cols, n_ids_index, n_ids, Batches, n_id_map, false, sample_size);
        });
    });
    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([&] { work_by_thread(matrix1, linkedLists, cols, n_ids_index, n_ids, Batches, n_id_map, sample_size); });
    }


    std::thread submission_thread(io_submission_thread, null_blocks); 

    wait_for_comp(null_blocks.size());

    submission_thread.join();
    for (auto& worker : workers) {
        worker.join();
    }
    
} 


void before_iteration_by_thread(std::vector<std::vector<int64_t>>& Batches, 
                                std::vector<std::vector<std::vector<int64_t>>>& cols, 
                                std::vector<std::vector<int64_t>>& n_ids, 
                                std::vector<tbb::concurrent_hash_map<int64_t, int>>& n_id_map, 
                                std::vector<std::vector<ListNode>>& linkedLists, 
                                std::vector<std::vector<MatrixEntry>>& matrix1, 
                                std::vector<int>& nextFreeSlots, 
                                int i, int iteration) {
    nextFreeSlots[i] = 0;
    matrix1[i].assign(size_index_out, {-1,-1});
    int64_t j;
    size_t batch_size = Batches[i].size();
    linkedLists[i].assign(batch_size, {nullptr, -1}); 
    cols[i].resize(batch_size); 
    
    for (int n = 0 ; n < batch_size ; n++) {
        j = Batches[i][n];
        n_ids[i][n] = j;
        if(iteration == 0) {
            tbb::concurrent_hash_map<int64_t, int>::accessor accessor;
            n_id_map[i].insert(accessor, j);
            accessor->second = n;
        }
    }
}

void before_iteration(int batchNum, 
                      std::vector<std::vector<int64_t>>& Batches, 
                      std::vector<std::vector<std::vector<int64_t>>>& cols, 
                      std::vector<std::vector<int64_t>>& n_ids, 
                      std::vector<tbb::concurrent_hash_map<int64_t, int>>& n_id_map, 
                      std::vector<std::vector<ListNode>>& linkedLists, 
                      std::vector<std::vector<MatrixEntry>>& matrix1, 
                      std::vector<int>& nextFreeSlots, 
                      std::vector<std::atomic<bool>>& block_indicator, int iteration) {

    tbb::parallel_for(0, batchNum, [&](int i) {
        before_iteration_by_thread(Batches, cols, n_ids, n_id_map, linkedLists, matrix1, nextFreeSlots, i, iteration);
    });

    tbb::parallel_for(0, size_index_out, [&](int i) {
        block_indicator[i].store(false);
    });
}

void after_iteration_by_thread(std::vector<std::vector<int64_t>>& Batches, 
                               std::vector<std::vector<std::vector<int64_t>>>& cols, 
                               std::vector<std::vector<int64_t>>& n_ids, 
                               std::vector<tbb::concurrent_hash_map<int64_t, int>>& n_id_map, 
                               std::vector<std::atomic<int>>& n_ids_index, 
                               std::vector<std::vector<ListNode>>& linkedLists, 
                               std::vector<std::vector<MatrixEntry>>& matrix1,
                               std::vector<torch::Tensor>& out_cols,
                               std::vector<torch::Tensor>& out_rowptrs,
                               std::vector<torch::Tensor>& out_n_ids,
                               std::vector<torch::Tensor>& batches,  
                               int i, int iteration) {
    
    auto out_rowptr = torch::empty(Batches[i].size()+1, batches[i].options());
    auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
    out_rowptr_data[0] = 0;
    int64_t E = 0;
    for (std::vector<int64_t> &col_vec : cols[i]) {
        E += col_vec.size();
    }
    num_total_act_node.fetch_add(E);
    auto out_col = torch::empty(E, batches[i].options());
    auto out_col_data = out_col.data_ptr<int64_t>();
    int col_i = 0;
    int rowptr_index = 1;
    for (std::vector<int64_t> &col_vec : cols[i]) {
        std::sort(col_vec.begin(), col_vec.end());
        std::copy(col_vec.begin(), col_vec.end(), out_col_data + col_i);
        col_i += col_vec.size();
        out_rowptr_data[rowptr_index] = col_i;
        rowptr_index++;
    }
    out_cols[i] = out_col;

    out_rowptrs[i] = out_rowptr;
    int64_t N = n_ids_index[i];
    auto out_n_id = torch::from_blob(n_ids[i].data(), {N}, batches[i].options()).clone();
    out_n_ids[i] = out_n_id;
    
    Batches[i].clear();
    std::vector<int64_t>().swap(Batches[i]);
    if(iteration != 2){
        Batches[i].reserve(n_ids_index[i]);
        Batches[i].insert(Batches[i].end(), n_ids[i].begin(), n_ids[i].begin() + n_ids_index[i]);
    }
    if(iteration != 2){
        n_ids[i].resize(2000000);
        n_ids[i].clear();
    }
    
    cols[i].clear(); 
    std::vector<std::vector<int64_t>>().swap(cols[i]); 

    if(iteration == 2){
        n_id_map[i].clear();
        n_ids[i].clear();
        linkedLists[i].clear();
        matrix1[i].clear();
        std::vector<int64_t>().swap(n_ids[i]); 
        n_id_map[i] = tbb::concurrent_hash_map<int64_t, int>();
        std::vector<MatrixEntry>().swap(matrix1[i]); 
        std::vector<ListNode>().swap(linkedLists[i]); 
    }
}

void after_iteration(int batchNum, 
                     std::vector<std::vector<int64_t>>& Batches, 
                     std::vector<std::vector<std::vector<int64_t>>>& cols, 
                     std::vector<std::vector<int64_t>>& n_ids,  
                     std::vector<tbb::concurrent_hash_map<int64_t, int>>& n_id_map, 
                     std::vector<std::atomic<int>>& n_ids_index, 
                     std::vector<std::vector<ListNode>>& linkedLists, 
                     std::vector<std::vector<MatrixEntry>>& matrix1, 
                     std::vector<torch::Tensor>& out_cols,
                     std::vector<torch::Tensor>& out_rowptrs,
                     std::vector<torch::Tensor>& out_n_ids,
                     std::vector<torch::Tensor>& batches,  
                     int iteration) {
    
    tbb::parallel_for(0, batchNum, [&](int i) {
        after_iteration_by_thread(Batches, cols, n_ids, n_id_map, n_ids_index, linkedLists, matrix1, out_cols, out_rowptrs, out_n_ids, batches, i, iteration);
    });
}



std::tuple<std::vector<std::vector<torch::Tensor>>, std::vector<std::vector<torch::Tensor>>, std::vector<std::vector<torch::Tensor>>>
bb_sample(std::string topology_path, std::vector<torch::Tensor> batches, std::vector<int64_t> num_neighbors, int64_t batch_size, std::string dataset, int buffer_size, int num_workers) { 
    auto start_whole = std::chrono::high_resolution_clock::now();
    create_basic_components(topology_path, dataset, buffer_size);
    
    
    int batchSize = batch_size;
    int batchNum = batches.size();
    int totalBlock = size_index_out;
    int hop = num_neighbors.size();
    
    std::vector<std::atomic<bool>> block_indicator(totalBlock); 
    std::vector<std::vector<int64_t>> Batches(batchNum); 

    std::vector<std::vector<int64_t>> n_ids(batchNum); 
    std::vector<tbb::concurrent_hash_map<int64_t, int>> n_id_map(batchNum); 
    std::vector<std::vector<std::vector<int64_t>>> cols(batchNum); 
    std::vector<std::atomic<int>> n_ids_index(batchNum);

    std::vector<std::vector<torch::Tensor>> out_n_ids(hop, std::vector<torch::Tensor>(batchNum));
    std::vector<std::vector<torch::Tensor>> out_cols(hop, std::vector<torch::Tensor>(batchNum));
    std::vector<std::vector<torch::Tensor>> out_rowptrs(hop, std::vector<torch::Tensor>(batchNum)); 
    
    
    tbb::parallel_for(0, batchNum, [&](int i) {
        torch::Tensor contig_batch = batches[i].contiguous();
        auto data_ptr = contig_batch.data_ptr<int64_t>();
        Batches[i] = std::vector<int64_t>(data_ptr, data_ptr + contig_batch.numel());
        n_ids_index[i].store(batchSize);
        n_ids[i].resize(2000000); 
    });
    n_ids_index[1207].store(179);
    
    
    // 매트릭스, 링크드리스트 생성
    std::vector<std::vector<MatrixEntry>> matrix_for_head(batchNum, std::vector<MatrixEntry>(size_index_out, {-1, -1}));//배치 x 블록
    std::vector<std::vector<ListNode>> linkedLists(batchNum, std::vector<ListNode>(batchSize, {nullptr, -1})); //전체 사이즈 뒤로 뺌
    std::vector<int> nextFreeSlots(batchNum, 0);
    
    for (int iteration = 0 ; iteration < hop ; iteration++){
        
        //std::cout << "iteration" << iteration << std::endl;
        before_iteration(batchNum, Batches, cols, n_ids, n_id_map, linkedLists, matrix_for_head, nextFreeSlots, block_indicator, iteration);

        auto start = std::chrono::high_resolution_clock::now();
        tbb::parallel_for(0, batchNum, [&](int i) {
            make_linked_list(Batches[i], i, matrix_for_head[i], linkedLists[i], nextFreeSlots[i], block_indicator);
        });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        //std::cout << "Time taken for bucket creation: " << elapsed.count() << " seconds" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        process_by_block_async(matrix_for_head, linkedLists, block_indicator, cols, n_ids_index, n_ids, Batches, n_id_map, num_workers, num_neighbors[iteration]);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        //std::cout << "Time taken for sample: " << elapsed.count() << " seconds" << std::endl;
        
        after_iteration(batchNum, Batches, cols, n_ids, n_id_map, n_ids_index, linkedLists, matrix_for_head, out_cols[iteration], out_rowptrs[iteration], out_n_ids[iteration], batches, iteration);
    }
    
    //std::cout << "num_blocks_sample =" << total_num_blocks << std::endl;
    //std::cout << "total_num_node_sample =" << num_total_act_node << std::endl;

    destroy_basic_components();
    
    
    auto end_whole = std::chrono::high_resolution_clock::now();
    auto duration_whole = std::chrono::duration<double, std::milli>(end_whole - start_whole).count();
    std::cout << "Elapsedtime= " << duration_whole << " seconds" <<  std::endl;

    malloc_trim(0);

    return std::make_tuple(out_rowptrs, out_cols, out_n_ids);
}

int bb_gather(std::vector<torch::Tensor> out_n_ids, int maxValue, int feature_dim, int buffer_size, int num_workers, std::string feature_path) {
    int batchNum = out_n_ids.size();
    feature_q_index.assign(maxValue,0);    
    storage_q_index.assign(batchNum, std::vector<int64_t>());  
    batch_trashcan.assign(batchNum, std::vector<float>());    
    feature_GCS.assign(maxValue, 0); 
    trashcan_count.assign(batchNum, std::atomic<int>(0));

    
    std::vector<std::vector<int64_t>> Batches(batchNum);
    tbb::parallel_for(0, batchNum, [&](int i) {
        torch::Tensor contig_tensor = out_n_ids[i].contiguous();
        auto data_ptr = contig_tensor.data_ptr<int64_t>();
        int64_t num_elements = contig_tensor.numel();
        Batches[i] = std::vector<int64_t>(data_ptr, data_ptr + num_elements);
    });
    int flag = gather(Batches, maxValue, GPU_feature_q, CPU_feature_q, feature_q_index, storage_q_index, batch_trashcan, trashcan_count, feature_GCS, feature_dim, buffer_size, num_workers, feature_path); 
    GCS_flag = flag;
    return 0;
}

torch::Tensor gather_sagnn(std::string feature_file, torch::Tensor idx, int64_t feature_dim, int i, int num_workers){
    torch::Tensor contig_tensor = idx.contiguous();
    auto data_ptr = contig_tensor.data_ptr<int64_t>();
    int64_t num_elements = contig_tensor.numel();
 
    std::vector<int64_t> n_ids = std::vector<int64_t>(data_ptr, data_ptr + num_elements);
    int N = n_ids.size();
    int odd = i % 2;
    if (odd == 0){
        batch_feature_queue1.resize(feature_dim * N);
        batch_trashcan_queue.resize((trashcan_count[i]+10)*(BLOCKSIZE/sizeof(float)));
    
 
        gather_by_batch(n_ids, GPU_feature_q, CPU_feature_q, feature_q_index, storage_q_index[i], batch_trashcan, feature_GCS, batch_feature_queue1, batch_trashcan_queue, i, feature_dim, GCS_flag, trashcan_count[i], num_workers);
        
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);
        auto result = torch::from_blob(batch_feature_queue1.data(), {N, feature_dim}, options).clone();
        
        return result;
    }else {
        batch_feature_queue2.resize(feature_dim * N);
        batch_trashcan_queue.resize((trashcan_count[i]+10)*(BLOCKSIZE/sizeof(float)));
    
        
        gather_by_batch(n_ids, GPU_feature_q, CPU_feature_q, feature_q_index, storage_q_index[i], batch_trashcan, feature_GCS, batch_feature_queue2, batch_trashcan_queue, i, feature_dim, GCS_flag, trashcan_count[i], num_workers);
        
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);
        auto result = torch::from_blob(batch_feature_queue2.data(), {N, feature_dim}, options).clone();
        return result;
    }
}

void free_memory(std::vector<std::vector<torch::Tensor>>& tensor_vector) {
    for (auto& row : tensor_vector) {
        for (auto& tensor : row) {
            tensor.reset();  
        }
        row.clear();        
        row.shrink_to_fit(); 
    }
    tensor_vector.clear();  
    tensor_vector.shrink_to_fit();  
    malloc_trim(0);  
}



PYBIND11_MODULE(sample, m) {
	m.def("bb_sample", &bb_sample, "block_based_sample");
    m.def("bb_gather", &bb_gather, "block_based_gather");
    m.def("gather_sagnn", &gather_sagnn, "block_based_gather");
    m.def("create_basic_components", &create_basic_components, "create_buffer_components");
    m.def("destroy_basic_components", &destroy_basic_components, "distroy_buffer_components");
    m.def("free_memory", &free_memory, "free_memory");
    m.def("allocate_global_buffer", &allocate_global_buffer, "allocate_global_buffer");
    m.def("free_global_buffer", &free_global_buffer, "free_global_buffer");
}

