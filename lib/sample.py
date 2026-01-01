import torch
import os
from typing import List, NamedTuple, Optional, Tuple
from torch import Tensor
from torch_sparse import SparseTensor
from lib.cpp_extension.wrapper import sample 
import threading
import psutil
import gc

class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]


    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / (1024 ** 2):.2f} MB")  
    print(f"VMS: {mem_info.vms / (1024 ** 2):.2f} MB")  



def process_batch(b, batches, sizes, out_rowptrs, out_cols, out_n_ids, exp_name, sb): 
    adjs = []
    for h in range(len(sizes)):
        if h == 0:
            subset_size = batches[b].size(0)
        else:
            subset_size = out_n_ids[h - 1][b].size(0)

        adj_t = SparseTensor(rowptr=out_rowptrs[h][b], row=None, col=out_cols[h][b],
                             sparse_sizes=(subset_size, out_n_ids[h][b].size(0)),
                             is_sorted=True)
        e_id = adj_t.storage.value()
        size = adj_t.sizes()[::-1]
        adjs.append(Adj(adj_t, e_id, size))

    adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]

    n_id_filename = os.path.join('./trace', exp_name, 'sb_' + str(sb) + '_ids_' + str(b) + '.pth')
    adjs_filename = os.path.join('./trace', exp_name, 'sb_' + str(sb) + '_adjs_' + str(b) + '.pth')

    torch.save(out_n_ids[len(sizes) - 1][b], n_id_filename)
    torch.save(adjs, adjs_filename)


def block_based_sample(exp_name, sb, node_idx: Tensor, sizes: List[int], batch_size, topology_path, dataset, buffer_size, num_workers):
    batches = torch.split(node_idx, batch_size)

    out_rowptrs, out_cols, out_n_ids = sample.bb_sample(topology_path, batches, sizes, batch_size, dataset, buffer_size, num_workers)


    num_threads = 32
    threads = []
    num_batches = len(batches)
    def worker(thread_id):
        for b in range(thread_id, num_batches, num_threads):
            process_batch(b, batches, sizes, out_rowptrs, out_cols, out_n_ids, exp_name, sb)

    for t in range(num_threads):
        thread = threading.Thread(target=worker, args=(t,))
        threads.append(thread)
        thread.start()  

    for thread in threads:
        thread.join()


    return out_n_ids
            

def block_based_gather(out_n_ids: List[Tensor], num_nodes, num_features, buffer_size, num_workers, feature_path):
    sample.bb_gather(out_n_ids, num_nodes, num_features, buffer_size, num_workers, feature_path)

    
        
def gather_sagnn(feature_file, idx, feature_dim, i, num_workers):
    return sample.gather_sagnn(feature_file, idx, feature_dim, i, num_workers)


def alloc_global():
    sample.allocate_global_buffer()

def free_global():
    sample.free_global_buffer()
    
