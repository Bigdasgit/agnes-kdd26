import os
from torch.utils.cpp_extension import load

dir_path = os.path.dirname(os.path.realpath(__file__))

sample = load(name='sample', sources=[os.path.join(dir_path, 'block_based_sample.cpp')], extra_cflags=['-g', '-mavx512f', '-mavx512vl', '-mavx512bw','-Og', '-std=c++17', '-I'+ dir_path], extra_ldflags=['-lpthread','-ltbb', '-laio', '-lboost_system'])
free = load(name='free', sources=[os.path.join(dir_path, 'free.cpp')], extra_cflags=['-O2'])
