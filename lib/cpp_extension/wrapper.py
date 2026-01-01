import os
from torch.utils.cpp_extension import load

dir_path = os.path.dirname(os.path.realpath(__file__))

#sample = load(name='sample', sources=[os.path.join(dir_path, 'sample_block.cpp')], extra_cflags=['-fopenmp', '-O2', '-I'+ dir_path], extra_ldflags=['-lgomp','-lrt'])
#sample = load(name='sample', sources=[os.path.join(dir_path, 'sample.cpp')], extra_cflags=['-fopenmp', '-O2', '-I'+ dir_path], extra_ldflags=['-lgomp','-lrt'])
sample = load(name='sample', sources=[os.path.join(dir_path, 'block_based_sample.cpp')], extra_cflags=['-g', '-mavx512f', '-mavx512vl', '-mavx512bw','-O2', '-std=c++17', '-I'+ dir_path], extra_ldflags=['-lpthread','-ltbb', '-laio', '-lboost_system'])
#gather = load(name='gather', sources=[os.path.join(dir_path, 'gather.cpp')], extra_cflags=['-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt'])
#mt_load = load(name='mt_load', sources=[os.path.join(dir_path, 'mt_load.cpp')], extra_cflags=['-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt'])
#update = load(name='update', sources=[os.path.join(dir_path, 'update.cpp')], extra_cflags=['-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt'])
free = load(name='free', sources=[os.path.join(dir_path, 'free.cpp')], extra_cflags=['-O2'])
#buffer = load(name='buffer', sources=[os.path.join(dir_path, 'buffer.cpp')], extra_cflags=['-O2','-I'+ dir_path])

