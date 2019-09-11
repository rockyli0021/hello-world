
node_name=ARGV[0]
ip=ARGV[1]

models_64 = Array.new
models_64 << "vgg16"
models_64 << "resnet50"
models_64 << "resnet101"
models_64 << "resnet152"
models_64 << "inceptionv3"

models_128 = Array.new
models_128 << "vgg16"
models_128 << "resnet50"
models_128 << "resnet152"




batch_size = 64
gpu_nums = [1,4,8,16,32]

models_64.each do |m|
  gpu_nums.each do |n|
    puts "===========#{m}====#{n}=========="
    cmd = Array.new
    if n < 16
      for i in 0..n-1      
        cmd << "HIP_VISIBLE_DEVICES=#{i} HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network #{m} --iterations 100 --dist-backend nccl --world-size #{n} --distributed_dataparallel --dist-url=\"tcp://#{ip}:54321\" --batch-size=#{batch_size} --rank #{i} >& ./log/#{n}_gpu_rank_#{i}_bs#{batch_size}_#{node_name}.log &"
      end
    else
      for i in 0..7      
        cmd << "HIP_VISIBLE_DEVICES=#{i} HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network #{m} --iterations 100 --dist-backend nccl --world-size #{n} --distributed_dataparallel --dist-url=\"tcp://#{ip}:54321\" --batch-size=#{batch_size} --rank #{i} >& ./log/#{n}_gpu_rank_#{i}_bs#{batch_size}_#{node_name}.log &"
      end
    end
    puts cmd
    puts "===================================="
  end
end


batch_size = 128


models_128.each do |m|
  gpu_nums.each do |n|
    puts "===========#{m}====#{n}=========="
    cmd = Array.new
    if n < 16
      for i in 0..n-1      
        cmd << "HIP_VISIBLE_DEVICES=#{i} HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network #{m} --iterations 100 --dist-backend nccl --world-size #{n} --distributed_dataparallel --dist-url=\"tcp://#{ip}:54321\" --batch-size=#{batch_size} --rank #{i} >& ./log/#{n}_gpu_rank_#{i}_bs#{batch_size}_#{node_name}.log &"
      end
    else
      for i in 0..7      
        cmd << "HIP_VISIBLE_DEVICES=#{i} HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network #{m} --iterations 100 --dist-backend nccl --world-size #{n} --distributed_dataparallel --dist-url=\"tcp://#{ip}:54321\" --batch-size=#{batch_size} --rank #{i} >& ./log/#{n}_gpu_rank_#{i}_bs#{batch_size}_#{node_name}.log &"
      end
    end
    puts cmd
    puts "===================================="
  end
end


https://raw.githubusercontent.com/rockyli123/hello-world/master/test.rb

https://raw.githubusercontent.com/rockyli123/hello-world/master/test.rb



HIP_VISIBLE_DEVICES=0 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 0 >& ./log/16_gpu_rank_0_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=1 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 1 >& ./log/16_gpu_rank_1_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=2 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 2 >& ./log/16_gpu_rank_2_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=3 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 3 >& ./log/16_gpu_rank_3_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=4 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 4 >& ./log/16_gpu_rank_4_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=5 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 5 >& ./log/16_gpu_rank_5_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=6 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 6 >& ./log/16_gpu_rank_6_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=7 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 7 >& ./log/16_gpu_rank_7_bs64_node63_vgg16.log &


HIP_VISIBLE_DEVICES=0 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 0 >& ./log/16_gpu_rank_0_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=1 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 1 >& ./log/16_gpu_rank_1_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=2 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 2 >& ./log/16_gpu_rank_2_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=3 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 3 >& ./log/16_gpu_rank_3_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=4 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 4 >& ./log/16_gpu_rank_4_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=5 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 5 >& ./log/16_gpu_rank_5_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=6 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 6 >& ./log/16_gpu_rank_6_bs64_node63_vgg16.log &
HIP_VISIBLE_DEVICES=7 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_IB_DISABLE=0  python3.6 micro_benchmarking_pytorch.py --network vgg16 --iterations 100 --dist-backend nccl --world-size 16 --distributed_dataparallel --dist-url="tcp://10.10.121.63:54321" --batch-size=64 --rank 7 >& ./log/16_gpu_rank_7_bs64_node63_vgg16.log &


