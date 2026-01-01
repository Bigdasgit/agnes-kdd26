import argparse
import time
import os
import glob
import multiprocessing
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
import threading
from queue import Queue
from sage import SAGE,GCN,GAT
from lib.utils import *
from lib.data import *
from lib.sample import *
#JM
import time 
import copy
total_compute_time = 0

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=11)
argparser.add_argument('--batch-size', type=int, default=1208)
argparser.add_argument('--num-workers', type=int, default=16)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--feature-dim', type=int, default=1024)
argparser.add_argument('--buffer-size', type=int, default=4096)
argparser.add_argument('--dataset', type=str, default='Yahoo')
argparser.add_argument('--dataset_path', type=str, default='/home/nvme1/Yahoo1M_64bit/')
argparser.add_argument('--exp-name', type=str, default=None)
argparser.add_argument('--sizes', type=str, default='10,10,10')
argparser.add_argument('--sb-size', type=int, default='1024')
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--feature-cache-size', type=float, default=500000000)
argparser.add_argument('--trace-load-num-threads', type=int, default=4)
argparser.add_argument('--neigh-cache-size', type=int, default=45000000000)
argparser.add_argument('--verbose', dest='verbose', default=True, action='store_true')
argparser.add_argument('--train-only', dest='train_only', default=True, action='store_true')
args = argparser.parse_args()

# Set args/environment variables/path
dataset_name = args.dataset
dataset_path = args.dataset_path #수정
buffer_size = args.buffer_size
num_workers = args.num_workers
split_idx_path = os.path.join(dataset_path, 'split_idx.pth') #수정

# Prepare dataset
if args.verbose:
    tqdm.write('Preparing dataset...')
if args.exp_name is None:
    now = datetime.now()
    args.exp_name = now.strftime('%Y_%m_%d_%H_%M_%S')
os.makedirs(os.path.join('./trace', args.exp_name), exist_ok=True)
sizes = [int(size) for size in args.sizes.split(',')]
dataset = SADataset(path=dataset_path, split_idx_path=split_idx_path)

num_nodes = dataset.num_nodes
num_features = args.feature_dim
features = dataset.features_path
num_classes = dataset.num_classes
labels = dataset.get_labels()
out_n_ids =None
#글로벌 out_rowptr, out_col, out_n_ids 아니며는 이거 디스크 저장


if args.verbose:
    tqdm.write('Done!')

# Define model
device = torch.device('cuda:%d' % args.gpu)
torch.cuda.set_device(device)
if args.model == 'sage':
    model = SAGE(num_features, args.num_hiddens, num_classes, num_layers=len(sizes))
elif args.model == 'gcn':
    model = GCN(num_features, args.num_hiddens, num_classes, num_layers=len(sizes), 
                norm=True)
elif args.model == 'gat':
    model = GAT(num_features, args.num_hiddens, num_classes, num_layers=len(sizes),
                heads=4)
model = model.to(device)


def inspect(i, last, mode='train'):
    global out_n_ids
    # Same effect of `sysctl -w vm.drop_caches=1`
    with open('/proc/sys/vm/drop_caches', 'w') as stream:
        stream.write('1\n')

    if mode == 'train':
        node_idx = dataset.shuffled_train_idx#dataset.train_idx #
    elif mode == 'valid':
        node_idx = dataset.val_idx
    elif mode == 'test':
        node_idx = dataset.test_idx

    gather_worker = None
    # no gather when i == 0
    if i != 0:
        #게더링(out_n_ids로)
        #block_based_gather(out_n_ids = out_n_ids[len(sizes)-1], num_nodes = num_nodes) 
        gather_worker = threading.Thread(target=block_based_gather, args=(out_n_ids[len(sizes)-1], num_nodes, num_features, buffer_size, num_workers, dataset_path))
        gather_worker.start()
    
    if not last:
        start_idx = i * args.batch_size * args.sb_size
        end_idx = min((i + 1) * args.batch_size * args.sb_size, node_idx.numel())
        out_n_ids = block_based_sample(exp_name=args.exp_name, sb=i, node_idx=node_idx[start_idx:end_idx], sizes=sizes, batch_size=args.batch_size, topology_path=dataset_path, dataset = dataset_name, buffer_size = buffer_size, num_workers = num_workers)

    if gather_worker is not None:
        gather_worker.join()


def trace_load(q, indices, sb):
    for i in indices:
        q.put((
            #torch.load('/home/nvme1/Ginex/trace/2024_12_04_06_54_27/' + 'sb_' + str(sb) + '_ids_' + str(i) + '.pth'),
            #torch.load('/home/nvme1/Ginex/trace/2024_12_04_06_54_27/' + 'sb_' + str(sb) + '_adjs_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_ids_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_adjs_' + str(i) + '.pth'),
            ))

def delete_trace(i):
    n_id_filelist = glob.glob('./trace/' + args.exp_name + '/sb_' + str(i - 1) + '_ids_*')
    adjs_filelist = glob.glob('./trace/' + args.exp_name + '/sb_' + str(i - 1) + '_adjs_*')

    for n_id_file in n_id_filelist:
        try:
            os.remove(n_id_file)
        except:
            tqdm.write('Error while deleting file : ', n_id_file)

    for adjs_file in adjs_filelist:
        try:
            os.remove(adjs_file)
        except:
            tqdm.write('Error while deleting file : ', adjs_file)


def gather(gather_q, n_id, batch_size, idx):
    local_batch_inputs = gather_sagnn(features, n_id, num_features, idx, num_workers)
    local_batch_labels = labels[n_id[:batch_size]]
    gather_q.put((local_batch_inputs, local_batch_labels))

def execute(i, pbar, total_loss, total_correct, last, mode='train'):
    if last:
        if mode == 'train':
            num_iter = int((dataset.shuffled_train_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
        elif mode == 'valid':
            num_iter = int((dataset.val_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
        elif mode == 'test':
            num_iter = int((dataset.test_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
    else:
        num_iter = args.sb_size
    # Multi-threaded load of sets of (ids, adj, update)
    #디스크에서 가져오는 작업. 일단 보류
    
    q = list()
    loader = list()
    for t in range(args.trace_load_num_threads):
        q.append(Queue(maxsize=2))
        loader.append(threading.Thread(target=trace_load, args=(q[t], list(range(t, num_iter, args.trace_load_num_threads)), i-1), daemon=True))
        loader[t].start()
    global total_compute_time
    n_id_q = Queue(maxsize=2) #cpp에서 직접 가져오기
    adjs_q = Queue(maxsize=2) #이건 디스크에서 가져오기
    gather_q = Queue(maxsize=1) #이건 게더큐 
    
    for idx in range(num_iter):
        #torch.set_printoptions(profile="full")
        batch_size = args.batch_size
        if idx == 0:
            # Sample
            q_value = q[idx % args.trace_load_num_threads].get()
            if q_value:
                n_id, adjs = q_value
                batch_size = adjs[-1].size[1]
                n_id_q.put(n_id)
                adjs_q.put(adjs)

            

            #print(n_id)
            # Gather
            #batch_inputs = copy.deepcopy(gather_sagnn(features, n_id, num_features, idx, num_workers)) #여기서는 직접 가져오는 작업 각 자료형을 맞춰줘야함
            batch_inputs = gather_sagnn(features, n_id, num_features, idx, num_workers) #여기서는 직접 가져오는 작업 각 자료형을 맞춰줘야함
            
            batch_labels = labels[n_id[:batch_size]]
            
            #torch.set_printoptions(precision=7)
 
            #print("batch_inputs =", batch_inputs)
            #print("batch_inputs_size =", batch_inputs.shape)
            #print("idx =", idx)
            #breakpoint()
            
        
        if idx != 0:
            # Gather
            (batch_inputs, batch_labels) = gather_q.get()
            #print("batch_inputs =", batch_inputs)
            #print("batch_inputs_size =", batch_inputs.shape)
            #print("idx =", idx)
            #breakpoint()

        if idx != num_iter-1:
            # Sample
            q_value = q[(idx + 1) % args.trace_load_num_threads].get()
            if q_value:
                n_id, adjs = q_value
                batch_size = adjs[-1].size[1]
                n_id_q.put(n_id)
                adjs_q.put(adjs)
            
            #print("batch_inputs =", batch_inputs)
            #print("batch_inputs_size =", batch_inputs.shape)
            #print("idx =", idx)
            #breakpoint()

            # Gather
            gather_loader = threading.Thread(target=gather, args=(gather_q, n_id, batch_size, idx + 1), daemon=True)
            gather_loader.start()
            #print("batch_inputs =", batch_inputs)
            #print("batch_inputs_size =", batch_inputs.shape)
            #print("idx =", idx)
            #breakpoint()
        
        
        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs_host = adjs_q.get()
        adjs = [adj.to(device) for adj in adjs_host]

        #print(batch_inputs)
        #print(batch_labels)
        #print(adjs_host)

        #print(batch_inputs)
        #print(batch_labels)
        #print(adjs_host)
        
        start_compute = time.time()
        # Forward
        out = model(batch_inputs_cuda, adjs)
        loss = F.nll_loss(out, batch_labels_cuda.long())

        # Backward
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Free
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        #print("total_loss=", total_loss)
        #print("total_correct=", total_correct)
        #breakpoint()
        n_id = n_id_q.get()
        #print(n_id)
        del(n_id)
        del(adjs_host)
        del(batch_inputs)
        pbar.update(batch_size)
        end_compute = time.time()
    
        # 이번 반복에서 걸린 시간 계산
        elapsed_compute = end_compute - start_compute
        
        # 누적 시간에 추가
        total_compute_time += elapsed_compute
        

    return total_loss, total_correct


def train(epoch):
    model.train()
    start_last = time.time()
    dataset.make_new_shuffled_train_idx()
    num_iter = int((dataset.shuffled_train_idx.numel()+args.batch_size-1) / args.batch_size)
 
    pbar = tqdm(total=dataset.train_idx.numel(), position=0, leave=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    num_sb = int((dataset.train_idx.numel()+args.batch_size*args.sb_size-1)/(args.batch_size*args.sb_size))
    stop_last = time.time()
    tqdm.write('last time: {:.4f} s'.format((stop_last - start_last)))
    start_for = time.time()
    for i in range(num_sb + 1):
        start_sample = time.time()

        if args.verbose:
            tqdm.write ('Running {}th hyperbatch of total {} hyperbatches'.format(i, num_sb))
        # hyperbatch sample
        if args.verbose:
            tqdm.write ('Step 1: hyperbatch Sample')
        inspect(i, last=(i==num_sb), mode='train')

        if args.verbose:
            tqdm.write ('Step 1: Done')
    
        stop_sample = time.time()
        tqdm.write('sample + gather time: {:.4f} s'.format((stop_sample - start_sample)))

        if i == 0:
            continue

        start_main = time.time()
        # Main loop
        alloc_global()
        if args.verbose:
            tqdm.write ('Step 2: Main Loop')
        total_loss, total_correct = execute(i, pbar, total_loss, total_correct, last=(i==num_sb), mode='train')
        #print('acc =', total_correct / (1000*(i+1)), flush=True)
        #print(i)
        if args.verbose:
            tqdm.write ('Step 2: Done')
        stop_main = time.time()
        tqdm.write('main time: {:.4f} s'.format((stop_main - start_main)))
        free_global()
        # Delete obsolete runtime files
        #delete_trace(i)

    pbar.close()
    stop_for = time.time()
    tqdm.write('for time: {:.4f} s'.format((stop_for - start_for)))
    loss = total_loss / num_iter
    approx_acc = total_correct / dataset.train_idx.numel()
    

    return loss, approx_acc


@torch.no_grad()
def inference(mode='test'):
    model.eval()

    if mode == 'test':
        pbar = tqdm(total=dataset.test_idx.numel(), position=0, leave=True)
        num_sb = int((dataset.test_idx.numel()+args.batch_size*args.sb_size-1)/(args.batch_size*args.sb_size))
        num_iter = int((dataset.test_idx.numel()+args.batch_size-1) / args.batch_size)
    elif mode == 'valid':
        pbar = tqdm(total=dataset.val_idx.numel(), position=0, leave=True)
        num_sb = int((dataset.val_idx.numel()+args.batch_size*args.sb_size-1)/(args.batch_size*args.sb_size))
        num_iter = int((dataset.val_idx.numel()+args.batch_size-1) / args.batch_size)

    pbar.set_description('Evaluating')

    total_loss = total_correct = 0

    for i in range(num_sb + 1):
        if args.verbose:

            tqdm.write ('Running {}th hyperbatch of total {} hyperbatches'.format(i, num_sb))
        
        # hyperbatch sample
        if args.verbose:
            tqdm.write ('Step 1: hyperbatch Sample')
        inspect(i, last=(i==num_sb), mode=mode)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write ('Step 1: Done')

        if i == 0:
            continue

        alloc_global()    
        # Main loop
        if args.verbose:
            tqdm.write ('Step 3: Main Loop')
        total_loss, total_correct = execute(i, pbar, total_loss, total_correct, last=(i==num_sb), mode=mode)
        if args.verbose:
            tqdm.write ('Step 3: Done')
        free_global()
        # Delete obsolete runtime files
        delete_trace(i)

    pbar.close()

    loss = total_loss / num_iter
    if mode == 'test':
        approx_acc = total_correct / dataset.test_idx.numel()
    elif mode == 'valid':
        approx_acc = total_correct / dataset.val_idx.numel()

    return loss, approx_acc


if __name__=='__main__':
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best_val_acc = final_test_acc = 0
    for epoch in range(args.num_epochs):
        if args.verbose:
            tqdm.write('\n==============================')
            tqdm.write('Running Epoch {}...'.format(epoch))

        start = time.time()
        loss, acc = train(epoch)
        end = time.time()
        tqdm.write(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        tqdm.write('Epoch time: {:.4f} ms'.format((end - start) * 1000))

        if epoch > 3 and not args.train_only:
            val_loss, val_acc = inference(mode='valid')
            test_loss, test_acc = inference(mode='test')
            tqdm.write ('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(val_loss, val_acc, test_loss, test_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    #버퍼관리
    #sample.destroy_basic_components()
    print ("total_compute_time =", total_compute_time)
    if not args.train_only:
        tqdm.write('Final Test acc: {final_test_acc:.4f}')
