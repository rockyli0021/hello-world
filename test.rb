
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


./1_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./1_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./1_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./1_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.25167925119400025
./1_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 254.291919959136
./1_gpu_rank_0_bs64_node63_resnet50.log-
./1_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./1_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.25167925119400025
./1_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 254.291919959136
./4_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./4_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./4_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./4_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./4_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./4_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.29137699842453
./4_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 219.64671317930654
./4_gpu_rank_0_bs64_node63_resnet50.log-
./4_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 4
./4_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 256
./4_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.29137699842453
./4_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 878.5868527172262
./8_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./8_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./8_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./8_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./8_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./8_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.2859256148338318
./8_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 223.83444042673187
./8_gpu_rank_0_bs64_node63_resnet50.log-
./8_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 8
./8_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 512
./8_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.2859256148338318
./8_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 1790.675523413855
./8_gpu_rank_1_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./8_gpu_rank_1_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./8_gpu_rank_1_bs64_node63_resnet50.log---------This process: rank 1--------
./8_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 1
./8_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 64
./8_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.2844381284713745
./8_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 225.00499614432275
./8_gpu_rank_1_bs64_node63_resnet50.log-
./8_gpu_rank_1_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 8
./8_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 512
./8_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.2844381284713745
./8_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 1800.039969154582
./16_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./16_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./16_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.2858107137680054
./16_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 223.92442591200154
./16_gpu_rank_0_bs64_node63_resnet50.log-
./16_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 16
./16_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.2858107137680054
./16_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 3582.7908145920246
./16_gpu_rank_1_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_1_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_1_bs64_node63_resnet50.log---------This process: rank 1--------
./16_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 1
./16_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.2853314995765686
./16_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 224.30050693658384
./16_gpu_rank_1_bs64_node63_resnet50.log-
./16_gpu_rank_1_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 16
./16_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.2853314995765686
./16_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 3588.8081109853415
./16_gpu_rank_2_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_2_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_2_bs64_node63_resnet50.log---------This process: rank 2--------
./16_gpu_rank_2_bs64_node63_resnet50.log-Num devices: 1
./16_gpu_rank_2_bs64_node63_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_2_bs64_node63_resnet50.log-Time per mini-batch : 0.2861491894721985
./16_gpu_rank_2_bs64_node63_resnet50.log-Throughput [img/sec] : 223.65955366865742
./16_gpu_rank_2_bs64_node63_resnet50.log-
./16_gpu_rank_2_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_2_bs64_node63_resnet50.log-Num devices: 16
./16_gpu_rank_2_bs64_node63_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_2_bs64_node63_resnet50.log-Time per mini-batch : 0.2861491894721985
./16_gpu_rank_2_bs64_node63_resnet50.log-Throughput [img/sec] : 3578.552858698519
./16_gpu_rank_3_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_3_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_3_bs64_node63_resnet50.log---------This process: rank 3--------
./16_gpu_rank_3_bs64_node63_resnet50.log-Num devices: 1
./16_gpu_rank_3_bs64_node63_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_3_bs64_node63_resnet50.log-Time per mini-batch : 0.2866872477531433
./16_gpu_rank_3_bs64_node63_resnet50.log-Throughput [img/sec] : 223.23978656737546
./16_gpu_rank_3_bs64_node63_resnet50.log-
./16_gpu_rank_3_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_3_bs64_node63_resnet50.log-Num devices: 16
./16_gpu_rank_3_bs64_node63_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_3_bs64_node63_resnet50.log-Time per mini-batch : 0.2866872477531433
./16_gpu_rank_3_bs64_node63_resnet50.log-Throughput [img/sec] : 3571.8365850780074
./32_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./32_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.28891496181488036
./32_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 221.51846895699165
./32_gpu_rank_0_bs64_node63_resnet50.log-
./32_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.28891496181488036
./32_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 7088.591006623733
./32_gpu_rank_1_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_1_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_1_bs64_node63_resnet50.log---------This process: rank 1--------
./32_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.28842978477478026
./32_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 221.89109231549804
./32_gpu_rank_1_bs64_node63_resnet50.log-
./32_gpu_rank_1_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.28842978477478026
./32_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 7100.514954095937
./32_gpu_rank_2_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_2_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_2_bs64_node63_resnet50.log---------This process: rank 2--------
./32_gpu_rank_2_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_2_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_2_bs64_node63_resnet50.log-Time per mini-batch : 0.28919888496398927
./32_gpu_rank_2_bs64_node63_resnet50.log-Throughput [img/sec] : 221.30099155800414
./32_gpu_rank_2_bs64_node63_resnet50.log-
./32_gpu_rank_2_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_2_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_2_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_2_bs64_node63_resnet50.log-Time per mini-batch : 0.28919888496398927
./32_gpu_rank_2_bs64_node63_resnet50.log-Throughput [img/sec] : 7081.631729856133
./32_gpu_rank_3_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_3_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_3_bs64_node63_resnet50.log---------This process: rank 3--------
./32_gpu_rank_3_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_3_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_3_bs64_node63_resnet50.log-Time per mini-batch : 0.2871611499786377
./32_gpu_rank_3_bs64_node63_resnet50.log-Throughput [img/sec] : 222.87137380791606
./32_gpu_rank_3_bs64_node63_resnet50.log-
./32_gpu_rank_3_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_3_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_3_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_3_bs64_node63_resnet50.log-Time per mini-batch : 0.2871611499786377
./32_gpu_rank_3_bs64_node63_resnet50.log-Throughput [img/sec] : 7131.883961853314
./32_gpu_rank_4_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_4_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_4_bs64_node63_resnet50.log---------This process: rank 4--------
./32_gpu_rank_4_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_4_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_4_bs64_node63_resnet50.log-Time per mini-batch : 0.28835497379302977
./32_gpu_rank_4_bs64_node63_resnet50.log-Throughput [img/sec] : 221.94865986926504
./32_gpu_rank_4_bs64_node63_resnet50.log-
./32_gpu_rank_4_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_4_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_4_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_4_bs64_node63_resnet50.log-Time per mini-batch : 0.28835497379302977
./32_gpu_rank_4_bs64_node63_resnet50.log-Throughput [img/sec] : 7102.357115816481
./32_gpu_rank_5_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_5_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_5_bs64_node63_resnet50.log---------This process: rank 5--------
./32_gpu_rank_5_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_5_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_5_bs64_node63_resnet50.log-Time per mini-batch : 0.29025033235549924
./32_gpu_rank_5_bs64_node63_resnet50.log-Throughput [img/sec] : 220.49931685043742
./32_gpu_rank_5_bs64_node63_resnet50.log-
./32_gpu_rank_5_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_5_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_5_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_5_bs64_node63_resnet50.log-Time per mini-batch : 0.29025033235549924
./32_gpu_rank_5_bs64_node63_resnet50.log-Throughput [img/sec] : 7055.978139213998
./32_gpu_rank_6_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_6_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_6_bs64_node63_resnet50.log---------This process: rank 6--------
./32_gpu_rank_6_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_6_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_6_bs64_node63_resnet50.log-Time per mini-batch : 0.29066019535064697
./32_gpu_rank_6_bs64_node63_resnet50.log-Throughput [img/sec] : 220.18838844717493
./32_gpu_rank_6_bs64_node63_resnet50.log-
./32_gpu_rank_6_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_6_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_6_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_6_bs64_node63_resnet50.log-Time per mini-batch : 0.29066019535064697
./32_gpu_rank_6_bs64_node63_resnet50.log-Throughput [img/sec] : 7046.028430309598
./32_gpu_rank_7_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_7_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_7_bs64_node63_resnet50.log---------This process: rank 7--------
./32_gpu_rank_7_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_7_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_7_bs64_node63_resnet50.log-Time per mini-batch : 0.2907558941841126
./32_gpu_rank_7_bs64_node63_resnet50.log-Throughput [img/sec] : 220.11591606625828
./32_gpu_rank_7_bs64_node63_resnet50.log-
./32_gpu_rank_7_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_7_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_7_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_7_bs64_node63_resnet50.log-Time per mini-batch : 0.2907558941841126
./32_gpu_rank_7_bs64_node63_resnet50.log-Throughput [img/sec] : 7043.709314120265
./1_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./1_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./1_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./1_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.4209385371208191
./1_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 152.04119926332743
./1_gpu_rank_0_bs64_node63_resnet101.log-
./1_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./1_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.4209385371208191
./1_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 152.04119926332743
./4_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./4_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./4_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./4_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./4_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./4_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.48895353078842163
./4_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 130.89178412681076
./4_gpu_rank_0_bs64_node63_resnet101.log-
./4_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 4
./4_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 256
./4_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.48895353078842163
./4_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 523.567136507243
./8_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./8_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./8_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./8_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./8_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./8_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.4807837200164795
./8_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 133.1159881990312
./8_gpu_rank_0_bs64_node63_resnet101.log-
./8_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 8
./8_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 512
./8_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.4807837200164795
./8_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 1064.9279055922495
./8_gpu_rank_1_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./8_gpu_rank_1_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./8_gpu_rank_1_bs64_node63_resnet101.log---------This process: rank 1--------
./8_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 1
./8_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 64
./8_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.4794627571105957
./8_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 133.48273468764413
./8_gpu_rank_1_bs64_node63_resnet101.log-
./8_gpu_rank_1_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 8
./8_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 512
./8_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.4794627571105957
./8_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 1067.861877501153
./16_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./16_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./16_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.49645888328552246
./16_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 128.9129919006655
./16_gpu_rank_0_bs64_node63_resnet101.log-
./16_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 16
./16_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.49645888328552246
./16_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 2062.607870410648
./16_gpu_rank_1_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_1_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_1_bs64_node63_resnet101.log---------This process: rank 1--------
./16_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 1
./16_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.4947663235664368
./16_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 129.35399389891202
./16_gpu_rank_1_bs64_node63_resnet101.log-
./16_gpu_rank_1_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 16
./16_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.4947663235664368
./16_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 2069.6639023825924
./16_gpu_rank_2_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_2_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_2_bs64_node63_resnet101.log---------This process: rank 2--------
./16_gpu_rank_2_bs64_node63_resnet101.log-Num devices: 1
./16_gpu_rank_2_bs64_node63_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_2_bs64_node63_resnet101.log-Time per mini-batch : 0.49207827568054197
./16_gpu_rank_2_bs64_node63_resnet101.log-Throughput [img/sec] : 130.06060857185435
./16_gpu_rank_2_bs64_node63_resnet101.log-
./16_gpu_rank_2_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_2_bs64_node63_resnet101.log-Num devices: 16
./16_gpu_rank_2_bs64_node63_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_2_bs64_node63_resnet101.log-Time per mini-batch : 0.49207827568054197
./16_gpu_rank_2_bs64_node63_resnet101.log-Throughput [img/sec] : 2080.9697371496695
./16_gpu_rank_3_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_3_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_3_bs64_node63_resnet101.log---------This process: rank 3--------
./16_gpu_rank_3_bs64_node63_resnet101.log-Num devices: 1
./16_gpu_rank_3_bs64_node63_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_3_bs64_node63_resnet101.log-Time per mini-batch : 0.49566025018692017
./16_gpu_rank_3_bs64_node63_resnet101.log-Throughput [img/sec] : 129.1207030942359
./16_gpu_rank_3_bs64_node63_resnet101.log-
./16_gpu_rank_3_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_3_bs64_node63_resnet101.log-Num devices: 16
./16_gpu_rank_3_bs64_node63_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_3_bs64_node63_resnet101.log-Time per mini-batch : 0.49566025018692017
./16_gpu_rank_3_bs64_node63_resnet101.log-Throughput [img/sec] : 2065.9312495077743
./32_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./32_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.49220637321472166
./32_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 130.02676008033004
./32_gpu_rank_0_bs64_node63_resnet101.log-
./32_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.49220637321472166
./32_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 4160.856322570561
./32_gpu_rank_1_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_1_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_1_bs64_node63_resnet101.log---------This process: rank 1--------
./32_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.49203326225280763
./32_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 130.0725071044418
./32_gpu_rank_1_bs64_node63_resnet101.log-
./32_gpu_rank_1_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.49203326225280763
./32_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 4162.3202273421375
./32_gpu_rank_2_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_2_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_2_bs64_node63_resnet101.log---------This process: rank 2--------
./32_gpu_rank_2_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_2_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_2_bs64_node63_resnet101.log-Time per mini-batch : 0.49179730892181395
./32_gpu_rank_2_bs64_node63_resnet101.log-Throughput [img/sec] : 130.13491297931185
./32_gpu_rank_2_bs64_node63_resnet101.log-
./32_gpu_rank_2_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_2_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_2_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_2_bs64_node63_resnet101.log-Time per mini-batch : 0.49179730892181395
./32_gpu_rank_2_bs64_node63_resnet101.log-Throughput [img/sec] : 4164.317215337979
./32_gpu_rank_3_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_3_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_3_bs64_node63_resnet101.log---------This process: rank 3--------
./32_gpu_rank_3_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_3_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_3_bs64_node63_resnet101.log-Time per mini-batch : 0.4890114164352417
./32_gpu_rank_3_bs64_node63_resnet101.log-Throughput [img/sec] : 130.8762901008372
./32_gpu_rank_3_bs64_node63_resnet101.log-
./32_gpu_rank_3_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_3_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_3_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_3_bs64_node63_resnet101.log-Time per mini-batch : 0.4890114164352417
./32_gpu_rank_3_bs64_node63_resnet101.log-Throughput [img/sec] : 4188.04128322679
./32_gpu_rank_4_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_4_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_4_bs64_node63_resnet101.log---------This process: rank 4--------
./32_gpu_rank_4_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_4_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_4_bs64_node63_resnet101.log-Time per mini-batch : 0.48815528869628905
./32_gpu_rank_4_bs64_node63_resnet101.log-Throughput [img/sec] : 131.10582120481394
./32_gpu_rank_4_bs64_node63_resnet101.log-
./32_gpu_rank_4_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_4_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_4_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_4_bs64_node63_resnet101.log-Time per mini-batch : 0.48815528869628905
./32_gpu_rank_4_bs64_node63_resnet101.log-Throughput [img/sec] : 4195.386278554046
./32_gpu_rank_5_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_5_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_5_bs64_node63_resnet101.log---------This process: rank 5--------
./32_gpu_rank_5_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_5_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_5_bs64_node63_resnet101.log-Time per mini-batch : 0.48794334173202514
./32_gpu_rank_5_bs64_node63_resnet101.log-Throughput [img/sec] : 131.16276937568773
./32_gpu_rank_5_bs64_node63_resnet101.log-
./32_gpu_rank_5_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_5_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_5_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_5_bs64_node63_resnet101.log-Time per mini-batch : 0.48794334173202514
./32_gpu_rank_5_bs64_node63_resnet101.log-Throughput [img/sec] : 4197.208620022007
./32_gpu_rank_6_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_6_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_6_bs64_node63_resnet101.log---------This process: rank 6--------
./32_gpu_rank_6_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_6_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_6_bs64_node63_resnet101.log-Time per mini-batch : 0.49051752805709836
./32_gpu_rank_6_bs64_node63_resnet101.log-Throughput [img/sec] : 130.47444044150473
./32_gpu_rank_6_bs64_node63_resnet101.log-
./32_gpu_rank_6_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_6_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_6_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_6_bs64_node63_resnet101.log-Time per mini-batch : 0.49051752805709836
./32_gpu_rank_6_bs64_node63_resnet101.log-Throughput [img/sec] : 4175.182094128151
./32_gpu_rank_7_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_7_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_7_bs64_node63_resnet101.log---------This process: rank 7--------
./32_gpu_rank_7_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_7_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_7_bs64_node63_resnet101.log-Time per mini-batch : 0.4919735026359558
./32_gpu_rank_7_bs64_node63_resnet101.log-Throughput [img/sec] : 130.0883069049308
./32_gpu_rank_7_bs64_node63_resnet101.log-
./32_gpu_rank_7_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_7_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_7_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_7_bs64_node63_resnet101.log-Time per mini-batch : 0.4919735026359558
./32_gpu_rank_7_bs64_node63_resnet101.log-Throughput [img/sec] : 4162.825820957785


  
  
  ====================================
    
  ./1_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./1_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./1_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./1_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.25167925119400025
./1_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 254.291919959136
./1_gpu_rank_0_bs64_node63_resnet50.log-
./1_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./1_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.25167925119400025
./1_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 254.291919959136
./4_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./4_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./4_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./4_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./4_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./4_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.29137699842453
./4_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 219.64671317930654
./4_gpu_rank_0_bs64_node63_resnet50.log-
./4_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 4
./4_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 256
./4_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.29137699842453
./4_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 878.5868527172262
./8_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./8_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./8_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./8_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./8_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./8_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.2859256148338318
./8_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 223.83444042673187
./8_gpu_rank_0_bs64_node63_resnet50.log-
./8_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 8
./8_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 512
./8_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.2859256148338318
./8_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 1790.675523413855
./8_gpu_rank_1_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./8_gpu_rank_1_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./8_gpu_rank_1_bs64_node63_resnet50.log---------This process: rank 1--------
./8_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 1
./8_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 64
./8_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.2844381284713745
./8_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 225.00499614432275
./8_gpu_rank_1_bs64_node63_resnet50.log-
./8_gpu_rank_1_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 8
./8_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 512
./8_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.2844381284713745
./8_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 1800.039969154582
./16_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./16_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./16_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.2858107137680054
./16_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 223.92442591200154
./16_gpu_rank_0_bs64_node63_resnet50.log-
./16_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 16
./16_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.2858107137680054
./16_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 3582.7908145920246
./16_gpu_rank_1_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_1_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_1_bs64_node63_resnet50.log---------This process: rank 1--------
./16_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 1
./16_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.2853314995765686
./16_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 224.30050693658384
./16_gpu_rank_1_bs64_node63_resnet50.log-
./16_gpu_rank_1_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 16
./16_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.2853314995765686
./16_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 3588.8081109853415
./16_gpu_rank_2_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_2_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_2_bs64_node63_resnet50.log---------This process: rank 2--------
./16_gpu_rank_2_bs64_node63_resnet50.log-Num devices: 1
./16_gpu_rank_2_bs64_node63_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_2_bs64_node63_resnet50.log-Time per mini-batch : 0.2861491894721985
./16_gpu_rank_2_bs64_node63_resnet50.log-Throughput [img/sec] : 223.65955366865742
./16_gpu_rank_2_bs64_node63_resnet50.log-
./16_gpu_rank_2_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_2_bs64_node63_resnet50.log-Num devices: 16
./16_gpu_rank_2_bs64_node63_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_2_bs64_node63_resnet50.log-Time per mini-batch : 0.2861491894721985
./16_gpu_rank_2_bs64_node63_resnet50.log-Throughput [img/sec] : 3578.552858698519
./16_gpu_rank_3_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_3_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_3_bs64_node63_resnet50.log---------This process: rank 3--------
./16_gpu_rank_3_bs64_node63_resnet50.log-Num devices: 1
./16_gpu_rank_3_bs64_node63_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_3_bs64_node63_resnet50.log-Time per mini-batch : 0.2866872477531433
./16_gpu_rank_3_bs64_node63_resnet50.log-Throughput [img/sec] : 223.23978656737546
./16_gpu_rank_3_bs64_node63_resnet50.log-
./16_gpu_rank_3_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_3_bs64_node63_resnet50.log-Num devices: 16
./16_gpu_rank_3_bs64_node63_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_3_bs64_node63_resnet50.log-Time per mini-batch : 0.2866872477531433
./16_gpu_rank_3_bs64_node63_resnet50.log-Throughput [img/sec] : 3571.8365850780074
./32_gpu_rank_0_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_0_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_0_bs64_node63_resnet50.log---------This process: rank 0--------
./32_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.28891496181488036
./32_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 221.51846895699165
./32_gpu_rank_0_bs64_node63_resnet50.log-
./32_gpu_rank_0_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_0_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_0_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_0_bs64_node63_resnet50.log-Time per mini-batch : 0.28891496181488036
./32_gpu_rank_0_bs64_node63_resnet50.log-Throughput [img/sec] : 7088.591006623733
./32_gpu_rank_1_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_1_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_1_bs64_node63_resnet50.log---------This process: rank 1--------
./32_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.28842978477478026
./32_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 221.89109231549804
./32_gpu_rank_1_bs64_node63_resnet50.log-
./32_gpu_rank_1_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_1_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_1_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_1_bs64_node63_resnet50.log-Time per mini-batch : 0.28842978477478026
./32_gpu_rank_1_bs64_node63_resnet50.log-Throughput [img/sec] : 7100.514954095937
./32_gpu_rank_2_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_2_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_2_bs64_node63_resnet50.log---------This process: rank 2--------
./32_gpu_rank_2_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_2_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_2_bs64_node63_resnet50.log-Time per mini-batch : 0.28919888496398927
./32_gpu_rank_2_bs64_node63_resnet50.log-Throughput [img/sec] : 221.30099155800414
./32_gpu_rank_2_bs64_node63_resnet50.log-
./32_gpu_rank_2_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_2_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_2_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_2_bs64_node63_resnet50.log-Time per mini-batch : 0.28919888496398927
./32_gpu_rank_2_bs64_node63_resnet50.log-Throughput [img/sec] : 7081.631729856133
./32_gpu_rank_3_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_3_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_3_bs64_node63_resnet50.log---------This process: rank 3--------
./32_gpu_rank_3_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_3_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_3_bs64_node63_resnet50.log-Time per mini-batch : 0.2871611499786377
./32_gpu_rank_3_bs64_node63_resnet50.log-Throughput [img/sec] : 222.87137380791606
./32_gpu_rank_3_bs64_node63_resnet50.log-
./32_gpu_rank_3_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_3_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_3_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_3_bs64_node63_resnet50.log-Time per mini-batch : 0.2871611499786377
./32_gpu_rank_3_bs64_node63_resnet50.log-Throughput [img/sec] : 7131.883961853314
./32_gpu_rank_4_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_4_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_4_bs64_node63_resnet50.log---------This process: rank 4--------
./32_gpu_rank_4_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_4_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_4_bs64_node63_resnet50.log-Time per mini-batch : 0.28835497379302977
./32_gpu_rank_4_bs64_node63_resnet50.log-Throughput [img/sec] : 221.94865986926504
./32_gpu_rank_4_bs64_node63_resnet50.log-
./32_gpu_rank_4_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_4_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_4_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_4_bs64_node63_resnet50.log-Time per mini-batch : 0.28835497379302977
./32_gpu_rank_4_bs64_node63_resnet50.log-Throughput [img/sec] : 7102.357115816481
./32_gpu_rank_5_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_5_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_5_bs64_node63_resnet50.log---------This process: rank 5--------
./32_gpu_rank_5_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_5_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_5_bs64_node63_resnet50.log-Time per mini-batch : 0.29025033235549924
./32_gpu_rank_5_bs64_node63_resnet50.log-Throughput [img/sec] : 220.49931685043742
./32_gpu_rank_5_bs64_node63_resnet50.log-
./32_gpu_rank_5_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_5_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_5_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_5_bs64_node63_resnet50.log-Time per mini-batch : 0.29025033235549924
./32_gpu_rank_5_bs64_node63_resnet50.log-Throughput [img/sec] : 7055.978139213998
./32_gpu_rank_6_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_6_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_6_bs64_node63_resnet50.log---------This process: rank 6--------
./32_gpu_rank_6_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_6_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_6_bs64_node63_resnet50.log-Time per mini-batch : 0.29066019535064697
./32_gpu_rank_6_bs64_node63_resnet50.log-Throughput [img/sec] : 220.18838844717493
./32_gpu_rank_6_bs64_node63_resnet50.log-
./32_gpu_rank_6_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_6_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_6_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_6_bs64_node63_resnet50.log-Time per mini-batch : 0.29066019535064697
./32_gpu_rank_6_bs64_node63_resnet50.log-Throughput [img/sec] : 7046.028430309598
./32_gpu_rank_7_bs64_node63_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_7_bs64_node63_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_7_bs64_node63_resnet50.log---------This process: rank 7--------
./32_gpu_rank_7_bs64_node63_resnet50.log-Num devices: 1
./32_gpu_rank_7_bs64_node63_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_7_bs64_node63_resnet50.log-Time per mini-batch : 0.2907558941841126
./32_gpu_rank_7_bs64_node63_resnet50.log-Throughput [img/sec] : 220.11591606625828
./32_gpu_rank_7_bs64_node63_resnet50.log-
./32_gpu_rank_7_bs64_node63_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_7_bs64_node63_resnet50.log-Num devices: 32
./32_gpu_rank_7_bs64_node63_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_7_bs64_node63_resnet50.log-Time per mini-batch : 0.2907558941841126
./32_gpu_rank_7_bs64_node63_resnet50.log-Throughput [img/sec] : 7043.709314120265
./1_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./1_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./1_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./1_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.4209385371208191
./1_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 152.04119926332743
./1_gpu_rank_0_bs64_node63_resnet101.log-
./1_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./1_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.4209385371208191
./1_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 152.04119926332743
./4_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./4_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./4_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./4_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./4_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./4_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.48895353078842163
./4_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 130.89178412681076
./4_gpu_rank_0_bs64_node63_resnet101.log-
./4_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 4
./4_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 256
./4_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.48895353078842163
./4_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 523.567136507243
./8_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./8_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./8_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./8_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./8_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./8_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.4807837200164795
./8_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 133.1159881990312
./8_gpu_rank_0_bs64_node63_resnet101.log-
./8_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 8
./8_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 512
./8_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.4807837200164795
./8_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 1064.9279055922495
./8_gpu_rank_1_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./8_gpu_rank_1_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./8_gpu_rank_1_bs64_node63_resnet101.log---------This process: rank 1--------
./8_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 1
./8_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 64
./8_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.4794627571105957
./8_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 133.48273468764413
./8_gpu_rank_1_bs64_node63_resnet101.log-
./8_gpu_rank_1_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 8
./8_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 512
./8_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.4794627571105957
./8_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 1067.861877501153
./16_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./16_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./16_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.49645888328552246
./16_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 128.9129919006655
./16_gpu_rank_0_bs64_node63_resnet101.log-
./16_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 16
./16_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.49645888328552246
./16_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 2062.607870410648
./16_gpu_rank_1_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_1_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_1_bs64_node63_resnet101.log---------This process: rank 1--------
./16_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 1
./16_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.4947663235664368
./16_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 129.35399389891202
./16_gpu_rank_1_bs64_node63_resnet101.log-
./16_gpu_rank_1_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 16
./16_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.4947663235664368
./16_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 2069.6639023825924
./16_gpu_rank_2_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_2_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_2_bs64_node63_resnet101.log---------This process: rank 2--------
./16_gpu_rank_2_bs64_node63_resnet101.log-Num devices: 1
./16_gpu_rank_2_bs64_node63_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_2_bs64_node63_resnet101.log-Time per mini-batch : 0.49207827568054197
./16_gpu_rank_2_bs64_node63_resnet101.log-Throughput [img/sec] : 130.06060857185435
./16_gpu_rank_2_bs64_node63_resnet101.log-
./16_gpu_rank_2_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_2_bs64_node63_resnet101.log-Num devices: 16
./16_gpu_rank_2_bs64_node63_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_2_bs64_node63_resnet101.log-Time per mini-batch : 0.49207827568054197
./16_gpu_rank_2_bs64_node63_resnet101.log-Throughput [img/sec] : 2080.9697371496695
./16_gpu_rank_3_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_3_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_3_bs64_node63_resnet101.log---------This process: rank 3--------
./16_gpu_rank_3_bs64_node63_resnet101.log-Num devices: 1
./16_gpu_rank_3_bs64_node63_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_3_bs64_node63_resnet101.log-Time per mini-batch : 0.49566025018692017
./16_gpu_rank_3_bs64_node63_resnet101.log-Throughput [img/sec] : 129.1207030942359
./16_gpu_rank_3_bs64_node63_resnet101.log-
./16_gpu_rank_3_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_3_bs64_node63_resnet101.log-Num devices: 16
./16_gpu_rank_3_bs64_node63_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_3_bs64_node63_resnet101.log-Time per mini-batch : 0.49566025018692017
./16_gpu_rank_3_bs64_node63_resnet101.log-Throughput [img/sec] : 2065.9312495077743
./32_gpu_rank_0_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_0_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_0_bs64_node63_resnet101.log---------This process: rank 0--------
./32_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.49220637321472166
./32_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 130.02676008033004
./32_gpu_rank_0_bs64_node63_resnet101.log-
./32_gpu_rank_0_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_0_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_0_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_0_bs64_node63_resnet101.log-Time per mini-batch : 0.49220637321472166
./32_gpu_rank_0_bs64_node63_resnet101.log-Throughput [img/sec] : 4160.856322570561
./32_gpu_rank_1_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_1_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_1_bs64_node63_resnet101.log---------This process: rank 1--------
./32_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.49203326225280763
./32_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 130.0725071044418
./32_gpu_rank_1_bs64_node63_resnet101.log-
./32_gpu_rank_1_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_1_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_1_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_1_bs64_node63_resnet101.log-Time per mini-batch : 0.49203326225280763
./32_gpu_rank_1_bs64_node63_resnet101.log-Throughput [img/sec] : 4162.3202273421375
./32_gpu_rank_2_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_2_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_2_bs64_node63_resnet101.log---------This process: rank 2--------
./32_gpu_rank_2_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_2_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_2_bs64_node63_resnet101.log-Time per mini-batch : 0.49179730892181395
./32_gpu_rank_2_bs64_node63_resnet101.log-Throughput [img/sec] : 130.13491297931185
./32_gpu_rank_2_bs64_node63_resnet101.log-
./32_gpu_rank_2_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_2_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_2_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_2_bs64_node63_resnet101.log-Time per mini-batch : 0.49179730892181395
./32_gpu_rank_2_bs64_node63_resnet101.log-Throughput [img/sec] : 4164.317215337979
./32_gpu_rank_3_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_3_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_3_bs64_node63_resnet101.log---------This process: rank 3--------
./32_gpu_rank_3_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_3_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_3_bs64_node63_resnet101.log-Time per mini-batch : 0.4890114164352417
./32_gpu_rank_3_bs64_node63_resnet101.log-Throughput [img/sec] : 130.8762901008372
./32_gpu_rank_3_bs64_node63_resnet101.log-
./32_gpu_rank_3_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_3_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_3_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_3_bs64_node63_resnet101.log-Time per mini-batch : 0.4890114164352417
./32_gpu_rank_3_bs64_node63_resnet101.log-Throughput [img/sec] : 4188.04128322679
./32_gpu_rank_4_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_4_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_4_bs64_node63_resnet101.log---------This process: rank 4--------
./32_gpu_rank_4_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_4_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_4_bs64_node63_resnet101.log-Time per mini-batch : 0.48815528869628905
./32_gpu_rank_4_bs64_node63_resnet101.log-Throughput [img/sec] : 131.10582120481394
./32_gpu_rank_4_bs64_node63_resnet101.log-
./32_gpu_rank_4_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_4_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_4_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_4_bs64_node63_resnet101.log-Time per mini-batch : 0.48815528869628905
./32_gpu_rank_4_bs64_node63_resnet101.log-Throughput [img/sec] : 4195.386278554046
./32_gpu_rank_5_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_5_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_5_bs64_node63_resnet101.log---------This process: rank 5--------
./32_gpu_rank_5_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_5_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_5_bs64_node63_resnet101.log-Time per mini-batch : 0.48794334173202514
./32_gpu_rank_5_bs64_node63_resnet101.log-Throughput [img/sec] : 131.16276937568773
./32_gpu_rank_5_bs64_node63_resnet101.log-
./32_gpu_rank_5_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_5_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_5_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_5_bs64_node63_resnet101.log-Time per mini-batch : 0.48794334173202514
./32_gpu_rank_5_bs64_node63_resnet101.log-Throughput [img/sec] : 4197.208620022007
./32_gpu_rank_6_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_6_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_6_bs64_node63_resnet101.log---------This process: rank 6--------
./32_gpu_rank_6_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_6_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_6_bs64_node63_resnet101.log-Time per mini-batch : 0.49051752805709836
./32_gpu_rank_6_bs64_node63_resnet101.log-Throughput [img/sec] : 130.47444044150473
./32_gpu_rank_6_bs64_node63_resnet101.log-
./32_gpu_rank_6_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_6_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_6_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_6_bs64_node63_resnet101.log-Time per mini-batch : 0.49051752805709836
./32_gpu_rank_6_bs64_node63_resnet101.log-Throughput [img/sec] : 4175.182094128151
./32_gpu_rank_7_bs64_node63_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_7_bs64_node63_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_7_bs64_node63_resnet101.log---------This process: rank 7--------
./32_gpu_rank_7_bs64_node63_resnet101.log-Num devices: 1
./32_gpu_rank_7_bs64_node63_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_7_bs64_node63_resnet101.log-Time per mini-batch : 0.4919735026359558
./32_gpu_rank_7_bs64_node63_resnet101.log-Throughput [img/sec] : 130.0883069049308
./32_gpu_rank_7_bs64_node63_resnet101.log-
./32_gpu_rank_7_bs64_node63_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_7_bs64_node63_resnet101.log-Num devices: 32
./32_gpu_rank_7_bs64_node63_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_7_bs64_node63_resnet101.log-Time per mini-batch : 0.4919735026359558
./32_gpu_rank_7_bs64_node63_resnet101.log-Throughput [img/sec] : 4162.825820957785
./inceptionv4_1gpu.log:--------------------SUMMARY--------------------------
./inceptionv4_1gpu.log-Microbenchmark for network : inceptionv4
./inceptionv4_1gpu.log---------This process: rank 0--------
./inceptionv4_1gpu.log-Num devices: 1
./inceptionv4_1gpu.log-Mini batch size [img] : 64
./inceptionv4_1gpu.log-Time per mini-batch : 0.5938066172599793
./inceptionv4_1gpu.log-Throughput [img/sec] : 107.77919635742226
./inceptionv4_1gpu.log-
./inceptionv4_1gpu.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./inceptionv4_1gpu.log-Num devices: 1
./inceptionv4_1gpu.log-Mini batch size [img] : 64
./inceptionv4_1gpu.log-Time per mini-batch : 0.5938066172599793
./inceptionv4_1gpu.log-Throughput [img/sec] : 107.77919635742226
./inceptionv4_1gpu.log-
./1_gpu_rank_0_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./1_gpu_rank_0_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./1_gpu_rank_0_bs64_node63_resnet152.log---------This process: rank 0--------
./1_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.6016809678077698
./1_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 106.36866283669333
./1_gpu_rank_0_bs64_node63_resnet152.log-
./1_gpu_rank_0_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./1_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 1
./1_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 64
./1_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.6016809678077698
./1_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 106.36866283669333
./4_gpu_rank_0_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./4_gpu_rank_0_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./4_gpu_rank_0_bs64_node63_resnet152.log---------This process: rank 0--------
./4_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 1
./4_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 64
./4_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.6982614731788636
./4_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 91.65620968408498
./4_gpu_rank_0_bs64_node63_resnet152.log-
./4_gpu_rank_0_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 4
./4_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 256
./4_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.6982614731788636
./4_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 366.62483873633994
./8_gpu_rank_0_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./8_gpu_rank_0_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./8_gpu_rank_0_bs64_node63_resnet152.log---------This process: rank 0--------
./8_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 1
./8_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 64
./8_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.6957875680923462
./8_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 91.982097604115
./8_gpu_rank_0_bs64_node63_resnet152.log-
./8_gpu_rank_0_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 8
./8_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 512
./8_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.6957875680923462
./8_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 735.85678083292
./8_gpu_rank_1_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./8_gpu_rank_1_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./8_gpu_rank_1_bs64_node63_resnet152.log---------This process: rank 1--------
./8_gpu_rank_1_bs64_node63_resnet152.log-Num devices: 1
./8_gpu_rank_1_bs64_node63_resnet152.log-Mini batch size [img] : 64
./8_gpu_rank_1_bs64_node63_resnet152.log-Time per mini-batch : 0.6989891457557679
./8_gpu_rank_1_bs64_node63_resnet152.log-Throughput [img/sec] : 91.56079230787095
./8_gpu_rank_1_bs64_node63_resnet152.log-
./8_gpu_rank_1_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_1_bs64_node63_resnet152.log-Num devices: 8
./8_gpu_rank_1_bs64_node63_resnet152.log-Mini batch size [img] : 512
./8_gpu_rank_1_bs64_node63_resnet152.log-Time per mini-batch : 0.6989891457557679
./8_gpu_rank_1_bs64_node63_resnet152.log-Throughput [img/sec] : 732.4863384629676
./16_gpu_rank_0_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_0_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_0_bs64_node63_resnet152.log---------This process: rank 0--------
./16_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 1
./16_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.7000448203086853
./16_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 91.42271772224406
./16_gpu_rank_0_bs64_node63_resnet152.log-
./16_gpu_rank_0_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 16
./16_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.7000448203086853
./16_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 1462.763483555905
./16_gpu_rank_1_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_1_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_1_bs64_node63_resnet152.log---------This process: rank 1--------
./16_gpu_rank_1_bs64_node63_resnet152.log-Num devices: 1
./16_gpu_rank_1_bs64_node63_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_1_bs64_node63_resnet152.log-Time per mini-batch : 0.7004008960723876
./16_gpu_rank_1_bs64_node63_resnet152.log-Throughput [img/sec] : 91.37623946355644
./16_gpu_rank_1_bs64_node63_resnet152.log-
./16_gpu_rank_1_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_1_bs64_node63_resnet152.log-Num devices: 16
./16_gpu_rank_1_bs64_node63_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_1_bs64_node63_resnet152.log-Time per mini-batch : 0.7004008960723876
./16_gpu_rank_1_bs64_node63_resnet152.log-Throughput [img/sec] : 1462.0198314169031
./16_gpu_rank_2_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_2_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_2_bs64_node63_resnet152.log---------This process: rank 2--------
./16_gpu_rank_2_bs64_node63_resnet152.log-Num devices: 1
./16_gpu_rank_2_bs64_node63_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_2_bs64_node63_resnet152.log-Time per mini-batch : 0.698978669643402
./16_gpu_rank_2_bs64_node63_resnet152.log-Throughput [img/sec] : 91.562164597456
./16_gpu_rank_2_bs64_node63_resnet152.log-
./16_gpu_rank_2_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_2_bs64_node63_resnet152.log-Num devices: 16
./16_gpu_rank_2_bs64_node63_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_2_bs64_node63_resnet152.log-Time per mini-batch : 0.698978669643402
./16_gpu_rank_2_bs64_node63_resnet152.log-Throughput [img/sec] : 1464.994633559296
./16_gpu_rank_3_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_3_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_3_bs64_node63_resnet152.log---------This process: rank 3--------
./16_gpu_rank_3_bs64_node63_resnet152.log-Num devices: 1
./16_gpu_rank_3_bs64_node63_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_3_bs64_node63_resnet152.log-Time per mini-batch : 0.6972751617431641
./16_gpu_rank_3_bs64_node63_resnet152.log-Throughput [img/sec] : 91.78585945898631
./16_gpu_rank_3_bs64_node63_resnet152.log-
./16_gpu_rank_3_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_3_bs64_node63_resnet152.log-Num devices: 16
./16_gpu_rank_3_bs64_node63_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_3_bs64_node63_resnet152.log-Time per mini-batch : 0.6972751617431641
./16_gpu_rank_3_bs64_node63_resnet152.log-Throughput [img/sec] : 1468.573751343781
./32_gpu_rank_0_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_0_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_0_bs64_node63_resnet152.log---------This process: rank 0--------
./32_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 1
./32_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.6937701749801636
./32_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 92.24956953767852
./32_gpu_rank_0_bs64_node63_resnet152.log-
./32_gpu_rank_0_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_0_bs64_node63_resnet152.log-Num devices: 32
./32_gpu_rank_0_bs64_node63_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_0_bs64_node63_resnet152.log-Time per mini-batch : 0.6937701749801636
./32_gpu_rank_0_bs64_node63_resnet152.log-Throughput [img/sec] : 2951.986225205713
./32_gpu_rank_1_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_1_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_1_bs64_node63_resnet152.log---------This process: rank 1--------
./32_gpu_rank_1_bs64_node63_resnet152.log-Num devices: 1
./32_gpu_rank_1_bs64_node63_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_1_bs64_node63_resnet152.log-Time per mini-batch : 0.695508496761322
./32_gpu_rank_1_bs64_node63_resnet152.log-Throughput [img/sec] : 92.01900522857726
./32_gpu_rank_1_bs64_node63_resnet152.log-
./32_gpu_rank_1_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_1_bs64_node63_resnet152.log-Num devices: 32
./32_gpu_rank_1_bs64_node63_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_1_bs64_node63_resnet152.log-Time per mini-batch : 0.695508496761322
./32_gpu_rank_1_bs64_node63_resnet152.log-Throughput [img/sec] : 2944.608167314472
./32_gpu_rank_2_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_2_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_2_bs64_node63_resnet152.log---------This process: rank 2--------
./32_gpu_rank_2_bs64_node63_resnet152.log-Num devices: 1
./32_gpu_rank_2_bs64_node63_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_2_bs64_node63_resnet152.log-Time per mini-batch : 0.6955315589904785
./32_gpu_rank_2_bs64_node63_resnet152.log-Throughput [img/sec] : 92.01595408969233
./32_gpu_rank_2_bs64_node63_resnet152.log-
./32_gpu_rank_2_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_2_bs64_node63_resnet152.log-Num devices: 32
./32_gpu_rank_2_bs64_node63_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_2_bs64_node63_resnet152.log-Time per mini-batch : 0.6955315589904785
./32_gpu_rank_2_bs64_node63_resnet152.log-Throughput [img/sec] : 2944.5105308701545
./32_gpu_rank_3_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_3_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_3_bs64_node63_resnet152.log---------This process: rank 3--------
./32_gpu_rank_3_bs64_node63_resnet152.log-Num devices: 1
./32_gpu_rank_3_bs64_node63_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_3_bs64_node63_resnet152.log-Time per mini-batch : 0.696360399723053
./32_gpu_rank_3_bs64_node63_resnet152.log-Throughput [img/sec] : 91.90643239542801
./32_gpu_rank_3_bs64_node63_resnet152.log-
./32_gpu_rank_3_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_3_bs64_node63_resnet152.log-Num devices: 32
./32_gpu_rank_3_bs64_node63_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_3_bs64_node63_resnet152.log-Time per mini-batch : 0.696360399723053
./32_gpu_rank_3_bs64_node63_resnet152.log-Throughput [img/sec] : 2941.0058366536964
./32_gpu_rank_4_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_4_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_4_bs64_node63_resnet152.log---------This process: rank 4--------
./32_gpu_rank_4_bs64_node63_resnet152.log-Num devices: 1
./32_gpu_rank_4_bs64_node63_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_4_bs64_node63_resnet152.log-Time per mini-batch : 0.6960429310798645
./32_gpu_rank_4_bs64_node63_resnet152.log-Throughput [img/sec] : 91.9483513764133
./32_gpu_rank_4_bs64_node63_resnet152.log-
./32_gpu_rank_4_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_4_bs64_node63_resnet152.log-Num devices: 32
./32_gpu_rank_4_bs64_node63_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_4_bs64_node63_resnet152.log-Time per mini-batch : 0.6960429310798645
./32_gpu_rank_4_bs64_node63_resnet152.log-Throughput [img/sec] : 2942.3472440452256
./32_gpu_rank_5_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_5_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_5_bs64_node63_resnet152.log---------This process: rank 5--------
./32_gpu_rank_5_bs64_node63_resnet152.log-Num devices: 1
./32_gpu_rank_5_bs64_node63_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_5_bs64_node63_resnet152.log-Time per mini-batch : 0.6934325551986694
./32_gpu_rank_5_bs64_node63_resnet152.log-Throughput [img/sec] : 92.29448418622906
./32_gpu_rank_5_bs64_node63_resnet152.log-
./32_gpu_rank_5_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_5_bs64_node63_resnet152.log-Num devices: 32
./32_gpu_rank_5_bs64_node63_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_5_bs64_node63_resnet152.log-Time per mini-batch : 0.6934325551986694
./32_gpu_rank_5_bs64_node63_resnet152.log-Throughput [img/sec] : 2953.42349395933
./32_gpu_rank_6_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_6_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_6_bs64_node63_resnet152.log---------This process: rank 6--------
./32_gpu_rank_6_bs64_node63_resnet152.log-Num devices: 1
./32_gpu_rank_6_bs64_node63_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_6_bs64_node63_resnet152.log-Time per mini-batch : 0.6951551795005798
./32_gpu_rank_6_bs64_node63_resnet152.log-Throughput [img/sec] : 92.06577450229099
./32_gpu_rank_6_bs64_node63_resnet152.log-
./32_gpu_rank_6_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_6_bs64_node63_resnet152.log-Num devices: 32
./32_gpu_rank_6_bs64_node63_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_6_bs64_node63_resnet152.log-Time per mini-batch : 0.6951551795005798
./32_gpu_rank_6_bs64_node63_resnet152.log-Throughput [img/sec] : 2946.1047840733117
./32_gpu_rank_7_bs64_node63_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_7_bs64_node63_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_7_bs64_node63_resnet152.log---------This process: rank 7--------
./32_gpu_rank_7_bs64_node63_resnet152.log-Num devices: 1
./32_gpu_rank_7_bs64_node63_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_7_bs64_node63_resnet152.log-Time per mini-batch : 0.694277114868164
./32_gpu_rank_7_bs64_node63_resnet152.log-Throughput [img/sec] : 92.18221172701757
./32_gpu_rank_7_bs64_node63_resnet152.log-
./32_gpu_rank_7_bs64_node63_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_7_bs64_node63_resnet152.log-Num devices: 32
./32_gpu_rank_7_bs64_node63_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_7_bs64_node63_resnet152.log-Time per mini-batch : 0.694277114868164
./32_gpu_rank_7_bs64_node63_resnet152.log-Throughput [img/sec] : 2949.8307752645624

  
  ===========================
    
  ./16_gpu_rank_6_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_6_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_6_bs64_node64_resnet101.log---------This process: rank 6--------
./16_gpu_rank_6_bs64_node64_resnet101.log-Num devices: 1
./16_gpu_rank_6_bs64_node64_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_6_bs64_node64_resnet101.log-Time per mini-batch : 0.4966030693054199
./16_gpu_rank_6_bs64_node64_resnet101.log-Throughput [img/sec] : 128.87556270949835
./16_gpu_rank_6_bs64_node64_resnet101.log-
./16_gpu_rank_6_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_6_bs64_node64_resnet101.log-Num devices: 16
./16_gpu_rank_6_bs64_node64_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_6_bs64_node64_resnet101.log-Time per mini-batch : 0.4966030693054199
./16_gpu_rank_6_bs64_node64_resnet101.log-Throughput [img/sec] : 2062.0090033519737
./32_gpu_rank_12_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_12_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_12_bs64_node64_resnet101.log---------This process: rank 12--------
./32_gpu_rank_12_bs64_node64_resnet101.log-Num devices: 1
./32_gpu_rank_12_bs64_node64_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_12_bs64_node64_resnet101.log-Time per mini-batch : 0.4908550238609314
./32_gpu_rank_12_bs64_node64_resnet101.log-Throughput [img/sec] : 130.3847304986175
./32_gpu_rank_12_bs64_node64_resnet101.log-
./32_gpu_rank_12_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_12_bs64_node64_resnet101.log-Num devices: 32
./32_gpu_rank_12_bs64_node64_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_12_bs64_node64_resnet101.log-Time per mini-batch : 0.4908550238609314
./32_gpu_rank_12_bs64_node64_resnet101.log-Throughput [img/sec] : 4172.31137595576
./16_gpu_rank_4_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_4_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_4_bs64_node64_resnet50.log---------This process: rank 4--------
./16_gpu_rank_4_bs64_node64_resnet50.log-Num devices: 1
./16_gpu_rank_4_bs64_node64_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_4_bs64_node64_resnet50.log-Time per mini-batch : 0.2868179655075073
./16_gpu_rank_4_bs64_node64_resnet50.log-Throughput [img/sec] : 223.13804467148984
./16_gpu_rank_4_bs64_node64_resnet50.log-
./16_gpu_rank_4_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_4_bs64_node64_resnet50.log-Num devices: 16
./16_gpu_rank_4_bs64_node64_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_4_bs64_node64_resnet50.log-Time per mini-batch : 0.2868179655075073
./16_gpu_rank_4_bs64_node64_resnet50.log-Throughput [img/sec] : 3570.2087147438374
./32_gpu_rank_15_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_15_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_15_bs64_node64_resnet50.log---------This process: rank 15--------
./32_gpu_rank_15_bs64_node64_resnet50.log-Num devices: 1
./32_gpu_rank_15_bs64_node64_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_15_bs64_node64_resnet50.log-Time per mini-batch : 0.29178987503051756
./32_gpu_rank_15_bs64_node64_resnet50.log-Throughput [img/sec] : 219.33591764726725
./32_gpu_rank_15_bs64_node64_resnet50.log-
./32_gpu_rank_15_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_15_bs64_node64_resnet50.log-Num devices: 32
./32_gpu_rank_15_bs64_node64_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_15_bs64_node64_resnet50.log-Time per mini-batch : 0.29178987503051756
./32_gpu_rank_15_bs64_node64_resnet50.log-Throughput [img/sec] : 7018.749364712552
./32_gpu_rank_15_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_15_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_15_bs64_node64_resnet101.log---------This process: rank 15--------
./32_gpu_rank_15_bs64_node64_resnet101.log-Num devices: 1
./32_gpu_rank_15_bs64_node64_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_15_bs64_node64_resnet101.log-Time per mini-batch : 0.49258951902389525
./32_gpu_rank_15_bs64_node64_resnet101.log-Throughput [img/sec] : 129.9256227108141
./32_gpu_rank_15_bs64_node64_resnet101.log-
./32_gpu_rank_15_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_15_bs64_node64_resnet101.log-Num devices: 32
./32_gpu_rank_15_bs64_node64_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_15_bs64_node64_resnet101.log-Time per mini-batch : 0.49258951902389525
./32_gpu_rank_15_bs64_node64_resnet101.log-Throughput [img/sec] : 4157.6199267460515
./32_gpu_rank_8_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_8_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_8_bs64_node64_resnet50.log---------This process: rank 8--------
./32_gpu_rank_8_bs64_node64_resnet50.log-Num devices: 1
./32_gpu_rank_8_bs64_node64_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_8_bs64_node64_resnet50.log-Time per mini-batch : 0.291620717048645
./32_gpu_rank_8_bs64_node64_resnet50.log-Throughput [img/sec] : 219.46314599221088
./32_gpu_rank_8_bs64_node64_resnet50.log-
./32_gpu_rank_8_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_8_bs64_node64_resnet50.log-Num devices: 32
./32_gpu_rank_8_bs64_node64_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_8_bs64_node64_resnet50.log-Time per mini-batch : 0.291620717048645
./32_gpu_rank_8_bs64_node64_resnet50.log-Throughput [img/sec] : 7022.820671750748
./16_gpu_rank_5_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_5_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_5_bs64_node64_resnet152.log---------This process: rank 5--------
./16_gpu_rank_5_bs64_node64_resnet152.log-Num devices: 1
./16_gpu_rank_5_bs64_node64_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_5_bs64_node64_resnet152.log-Time per mini-batch : 0.7004351449012757
./16_gpu_rank_5_bs64_node64_resnet152.log-Throughput [img/sec] : 91.37177148502538
./16_gpu_rank_5_bs64_node64_resnet152.log-
./16_gpu_rank_5_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_5_bs64_node64_resnet152.log-Num devices: 16
./16_gpu_rank_5_bs64_node64_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_5_bs64_node64_resnet152.log-Time per mini-batch : 0.7004351449012757
./16_gpu_rank_5_bs64_node64_resnet152.log-Throughput [img/sec] : 1461.948343760406
./32_gpu_rank_12_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_12_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_12_bs64_node64_resnet50.log---------This process: rank 12--------
./32_gpu_rank_12_bs64_node64_resnet50.log-Num devices: 1
./32_gpu_rank_12_bs64_node64_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_12_bs64_node64_resnet50.log-Time per mini-batch : 0.28856935024261476
./32_gpu_rank_12_bs64_node64_resnet50.log-Throughput [img/sec] : 221.78377553330589
./32_gpu_rank_12_bs64_node64_resnet50.log-
./32_gpu_rank_12_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_12_bs64_node64_resnet50.log-Num devices: 32
./32_gpu_rank_12_bs64_node64_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_12_bs64_node64_resnet50.log-Time per mini-batch : 0.28856935024261476
./32_gpu_rank_12_bs64_node64_resnet50.log-Throughput [img/sec] : 7097.080817065788
./4_gpu_rank_1_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./4_gpu_rank_1_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./4_gpu_rank_1_bs64_node64_resnet101.log---------This process: rank 1--------
./4_gpu_rank_1_bs64_node64_resnet101.log-Num devices: 1
./4_gpu_rank_1_bs64_node64_resnet101.log-Mini batch size [img] : 64
./4_gpu_rank_1_bs64_node64_resnet101.log-Time per mini-batch : 0.4918073797225952
./4_gpu_rank_1_bs64_node64_resnet101.log-Throughput [img/sec] : 130.1322481905402
./4_gpu_rank_1_bs64_node64_resnet101.log-
./4_gpu_rank_1_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_1_bs64_node64_resnet101.log-Num devices: 4
./4_gpu_rank_1_bs64_node64_resnet101.log-Mini batch size [img] : 256
./4_gpu_rank_1_bs64_node64_resnet101.log-Time per mini-batch : 0.4918073797225952
./4_gpu_rank_1_bs64_node64_resnet101.log-Throughput [img/sec] : 520.5289927621608
./32_gpu_rank_11_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_11_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_11_bs64_node64_resnet101.log---------This process: rank 11--------
./32_gpu_rank_11_bs64_node64_resnet101.log-Num devices: 1
./32_gpu_rank_11_bs64_node64_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_11_bs64_node64_resnet101.log-Time per mini-batch : 0.490540931224823
./32_gpu_rank_11_bs64_node64_resnet101.log-Throughput [img/sec] : 130.46821564960854
./32_gpu_rank_11_bs64_node64_resnet101.log-
./32_gpu_rank_11_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_11_bs64_node64_resnet101.log-Num devices: 32
./32_gpu_rank_11_bs64_node64_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_11_bs64_node64_resnet101.log-Time per mini-batch : 0.490540931224823
./32_gpu_rank_11_bs64_node64_resnet101.log-Throughput [img/sec] : 4174.982900787473
./32_gpu_rank_14_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_14_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_14_bs64_node64_resnet50.log---------This process: rank 14--------
./32_gpu_rank_14_bs64_node64_resnet50.log-Num devices: 1
./32_gpu_rank_14_bs64_node64_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_14_bs64_node64_resnet50.log-Time per mini-batch : 0.29136894941329955
./32_gpu_rank_14_bs64_node64_resnet50.log-Throughput [img/sec] : 219.65278087754507
./32_gpu_rank_14_bs64_node64_resnet50.log-
./32_gpu_rank_14_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_14_bs64_node64_resnet50.log-Num devices: 32
./32_gpu_rank_14_bs64_node64_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_14_bs64_node64_resnet50.log-Time per mini-batch : 0.29136894941329955
./32_gpu_rank_14_bs64_node64_resnet50.log-Throughput [img/sec] : 7028.888988081442
./8_gpu_rank_2_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./8_gpu_rank_2_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./8_gpu_rank_2_bs64_node64_resnet50.log---------This process: rank 2--------
./8_gpu_rank_2_bs64_node64_resnet50.log-Num devices: 1
./8_gpu_rank_2_bs64_node64_resnet50.log-Mini batch size [img] : 64
./8_gpu_rank_2_bs64_node64_resnet50.log-Time per mini-batch : 0.287625629901886
./8_gpu_rank_2_bs64_node64_resnet50.log-Throughput [img/sec] : 222.51146402297837
./8_gpu_rank_2_bs64_node64_resnet50.log-
./8_gpu_rank_2_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_2_bs64_node64_resnet50.log-Num devices: 8
./8_gpu_rank_2_bs64_node64_resnet50.log-Mini batch size [img] : 512
./8_gpu_rank_2_bs64_node64_resnet50.log-Time per mini-batch : 0.287625629901886
./8_gpu_rank_2_bs64_node64_resnet50.log-Throughput [img/sec] : 1780.091712183827
./32_gpu_rank_11_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_11_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_11_bs64_node64_resnet50.log---------This process: rank 11--------
./32_gpu_rank_11_bs64_node64_resnet50.log-Num devices: 1
./32_gpu_rank_11_bs64_node64_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_11_bs64_node64_resnet50.log-Time per mini-batch : 0.29147753953933714
./32_gpu_rank_11_bs64_node64_resnet50.log-Throughput [img/sec] : 219.57094910691293
./32_gpu_rank_11_bs64_node64_resnet50.log-
./32_gpu_rank_11_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_11_bs64_node64_resnet50.log-Num devices: 32
./32_gpu_rank_11_bs64_node64_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_11_bs64_node64_resnet50.log-Time per mini-batch : 0.29147753953933714
./32_gpu_rank_11_bs64_node64_resnet50.log-Throughput [img/sec] : 7026.270371421214
./32_gpu_rank_13_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_13_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_13_bs64_node64_resnet152.log---------This process: rank 13--------
./32_gpu_rank_13_bs64_node64_resnet152.log-Num devices: 1
./32_gpu_rank_13_bs64_node64_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_13_bs64_node64_resnet152.log-Time per mini-batch : 0.6952860116958618
./32_gpu_rank_13_bs64_node64_resnet152.log-Throughput [img/sec] : 92.04845045551622
./32_gpu_rank_13_bs64_node64_resnet152.log-
./32_gpu_rank_13_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_13_bs64_node64_resnet152.log-Num devices: 32
./32_gpu_rank_13_bs64_node64_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_13_bs64_node64_resnet152.log-Time per mini-batch : 0.6952860116958618
./32_gpu_rank_13_bs64_node64_resnet152.log-Throughput [img/sec] : 2945.550414576519
./32_gpu_rank_13_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_13_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_13_bs64_node64_resnet101.log---------This process: rank 13--------
./32_gpu_rank_13_bs64_node64_resnet101.log-Num devices: 1
./32_gpu_rank_13_bs64_node64_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_13_bs64_node64_resnet101.log-Time per mini-batch : 0.49251237630844114
./32_gpu_rank_13_bs64_node64_resnet101.log-Throughput [img/sec] : 129.94597309351536
./32_gpu_rank_13_bs64_node64_resnet101.log-
./32_gpu_rank_13_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_13_bs64_node64_resnet101.log-Num devices: 32
./32_gpu_rank_13_bs64_node64_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_13_bs64_node64_resnet101.log-Time per mini-batch : 0.49251237630844114
./32_gpu_rank_13_bs64_node64_resnet101.log-Throughput [img/sec] : 4158.271138992492
./8_gpu_rank_2_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./8_gpu_rank_2_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./8_gpu_rank_2_bs64_node64_resnet152.log---------This process: rank 2--------
./8_gpu_rank_2_bs64_node64_resnet152.log-Num devices: 1
./8_gpu_rank_2_bs64_node64_resnet152.log-Mini batch size [img] : 64
./8_gpu_rank_2_bs64_node64_resnet152.log-Time per mini-batch : 0.6989349031448364
./8_gpu_rank_2_bs64_node64_resnet152.log-Throughput [img/sec] : 91.56789811473706
./8_gpu_rank_2_bs64_node64_resnet152.log-
./8_gpu_rank_2_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_2_bs64_node64_resnet152.log-Num devices: 8
./8_gpu_rank_2_bs64_node64_resnet152.log-Mini batch size [img] : 512
./8_gpu_rank_2_bs64_node64_resnet152.log-Time per mini-batch : 0.6989349031448364
./8_gpu_rank_2_bs64_node64_resnet152.log-Throughput [img/sec] : 732.5431849178965
./16_gpu_rank_7_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_7_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_7_bs64_node64_resnet50.log---------This process: rank 7--------
./16_gpu_rank_7_bs64_node64_resnet50.log-Num devices: 1
./16_gpu_rank_7_bs64_node64_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_7_bs64_node64_resnet50.log-Time per mini-batch : 0.28635382175445556
./16_gpu_rank_7_bs64_node64_resnet50.log-Throughput [img/sec] : 223.49972355137314
./16_gpu_rank_7_bs64_node64_resnet50.log-
./16_gpu_rank_7_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_7_bs64_node64_resnet50.log-Num devices: 16
./16_gpu_rank_7_bs64_node64_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_7_bs64_node64_resnet50.log-Time per mini-batch : 0.28635382175445556
./16_gpu_rank_7_bs64_node64_resnet50.log-Throughput [img/sec] : 3575.99557682197
./32_gpu_rank_10_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_10_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_10_bs64_node64_resnet152.log---------This process: rank 10--------
./32_gpu_rank_10_bs64_node64_resnet152.log-Num devices: 1
./32_gpu_rank_10_bs64_node64_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_10_bs64_node64_resnet152.log-Time per mini-batch : 0.6958671140670777
./32_gpu_rank_10_bs64_node64_resnet152.log-Throughput [img/sec] : 91.97158294483042
./32_gpu_rank_10_bs64_node64_resnet152.log-
./32_gpu_rank_10_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_10_bs64_node64_resnet152.log-Num devices: 32
./32_gpu_rank_10_bs64_node64_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_10_bs64_node64_resnet152.log-Time per mini-batch : 0.6958671140670777
./32_gpu_rank_10_bs64_node64_resnet152.log-Throughput [img/sec] : 2943.0906542345733
./4_gpu_rank_1_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./4_gpu_rank_1_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./4_gpu_rank_1_bs64_node64_resnet50.log---------This process: rank 1--------
./4_gpu_rank_1_bs64_node64_resnet50.log-Num devices: 1
./4_gpu_rank_1_bs64_node64_resnet50.log-Mini batch size [img] : 64
./4_gpu_rank_1_bs64_node64_resnet50.log-Time per mini-batch : 0.29138167142868043
./4_gpu_rank_1_bs64_node64_resnet50.log-Throughput [img/sec] : 219.64319061730984
./4_gpu_rank_1_bs64_node64_resnet50.log-
./4_gpu_rank_1_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_1_bs64_node64_resnet50.log-Num devices: 4
./4_gpu_rank_1_bs64_node64_resnet50.log-Mini batch size [img] : 256
./4_gpu_rank_1_bs64_node64_resnet50.log-Time per mini-batch : 0.29138167142868043
./4_gpu_rank_1_bs64_node64_resnet50.log-Throughput [img/sec] : 878.5727624692394
./16_gpu_rank_6_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_6_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_6_bs64_node64_resnet50.log---------This process: rank 6--------
./16_gpu_rank_6_bs64_node64_resnet50.log-Num devices: 1
./16_gpu_rank_6_bs64_node64_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_6_bs64_node64_resnet50.log-Time per mini-batch : 0.287432119846344
./16_gpu_rank_6_bs64_node64_resnet50.log-Throughput [img/sec] : 222.66126706442287
./16_gpu_rank_6_bs64_node64_resnet50.log-
./16_gpu_rank_6_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_6_bs64_node64_resnet50.log-Num devices: 16
./16_gpu_rank_6_bs64_node64_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_6_bs64_node64_resnet50.log-Time per mini-batch : 0.287432119846344
./16_gpu_rank_6_bs64_node64_resnet50.log-Throughput [img/sec] : 3562.580273030766
./32_gpu_rank_10_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_10_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_10_bs64_node64_resnet101.log---------This process: rank 10--------
./32_gpu_rank_10_bs64_node64_resnet101.log-Num devices: 1
./32_gpu_rank_10_bs64_node64_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_10_bs64_node64_resnet101.log-Time per mini-batch : 0.4920385909080505
./32_gpu_rank_10_bs64_node64_resnet101.log-Throughput [img/sec] : 130.07109845162526
./32_gpu_rank_10_bs64_node64_resnet101.log-
./32_gpu_rank_10_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_10_bs64_node64_resnet101.log-Num devices: 32
./32_gpu_rank_10_bs64_node64_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_10_bs64_node64_resnet101.log-Time per mini-batch : 0.4920385909080505
./32_gpu_rank_10_bs64_node64_resnet101.log-Throughput [img/sec] : 4162.275150452008
./8_gpu_rank_3_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./8_gpu_rank_3_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./8_gpu_rank_3_bs64_node64_resnet152.log---------This process: rank 3--------
./8_gpu_rank_3_bs64_node64_resnet152.log-Num devices: 1
./8_gpu_rank_3_bs64_node64_resnet152.log-Mini batch size [img] : 64
./8_gpu_rank_3_bs64_node64_resnet152.log-Time per mini-batch : 0.6977047538757324
./8_gpu_rank_3_bs64_node64_resnet152.log-Throughput [img/sec] : 91.72934489048785
./8_gpu_rank_3_bs64_node64_resnet152.log-
./8_gpu_rank_3_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_3_bs64_node64_resnet152.log-Num devices: 8
./8_gpu_rank_3_bs64_node64_resnet152.log-Mini batch size [img] : 512
./8_gpu_rank_3_bs64_node64_resnet152.log-Time per mini-batch : 0.6977047538757324
./8_gpu_rank_3_bs64_node64_resnet152.log-Throughput [img/sec] : 733.8347591239028
./16_gpu_rank_5_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_5_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_5_bs64_node64_resnet50.log---------This process: rank 5--------
./16_gpu_rank_5_bs64_node64_resnet50.log-Num devices: 1
./16_gpu_rank_5_bs64_node64_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_5_bs64_node64_resnet50.log-Time per mini-batch : 0.286319797039032
./16_gpu_rank_5_bs64_node64_resnet50.log-Throughput [img/sec] : 223.5262830647904
./16_gpu_rank_5_bs64_node64_resnet50.log-
./16_gpu_rank_5_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_5_bs64_node64_resnet50.log-Num devices: 16
./16_gpu_rank_5_bs64_node64_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_5_bs64_node64_resnet50.log-Time per mini-batch : 0.286319797039032
./16_gpu_rank_5_bs64_node64_resnet50.log-Throughput [img/sec] : 3576.4205290366463
./32_gpu_rank_9_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_9_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_9_bs64_node64_resnet50.log---------This process: rank 9--------
./32_gpu_rank_9_bs64_node64_resnet50.log-Num devices: 1
./32_gpu_rank_9_bs64_node64_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_9_bs64_node64_resnet50.log-Time per mini-batch : 0.2923352074623108
./32_gpu_rank_9_bs64_node64_resnet50.log-Throughput [img/sec] : 218.92676067165525
./32_gpu_rank_9_bs64_node64_resnet50.log-
./32_gpu_rank_9_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_9_bs64_node64_resnet50.log-Num devices: 32
./32_gpu_rank_9_bs64_node64_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_9_bs64_node64_resnet50.log-Time per mini-batch : 0.2923352074623108
./32_gpu_rank_9_bs64_node64_resnet50.log-Throughput [img/sec] : 7005.656341492968
./32_gpu_rank_14_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_14_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_14_bs64_node64_resnet152.log---------This process: rank 14--------
./32_gpu_rank_14_bs64_node64_resnet152.log-Num devices: 1
./32_gpu_rank_14_bs64_node64_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_14_bs64_node64_resnet152.log-Time per mini-batch : 0.6960262036323548
./32_gpu_rank_14_bs64_node64_resnet152.log-Throughput [img/sec] : 91.95056115129422
./32_gpu_rank_14_bs64_node64_resnet152.log-
./32_gpu_rank_14_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_14_bs64_node64_resnet152.log-Num devices: 32
./32_gpu_rank_14_bs64_node64_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_14_bs64_node64_resnet152.log-Time per mini-batch : 0.6960262036323548
./32_gpu_rank_14_bs64_node64_resnet152.log-Throughput [img/sec] : 2942.417956841415
./32_gpu_rank_13_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_13_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_13_bs64_node64_resnet50.log---------This process: rank 13--------
./32_gpu_rank_13_bs64_node64_resnet50.log-Num devices: 1
./32_gpu_rank_13_bs64_node64_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_13_bs64_node64_resnet50.log-Time per mini-batch : 0.2925617241859436
./32_gpu_rank_13_bs64_node64_resnet50.log-Throughput [img/sec] : 218.75725602206077
./32_gpu_rank_13_bs64_node64_resnet50.log-
./32_gpu_rank_13_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_13_bs64_node64_resnet50.log-Num devices: 32
./32_gpu_rank_13_bs64_node64_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_13_bs64_node64_resnet50.log-Time per mini-batch : 0.2925617241859436
./32_gpu_rank_13_bs64_node64_resnet50.log-Throughput [img/sec] : 7000.232192705945
./32_gpu_rank_9_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_9_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_9_bs64_node64_resnet152.log---------This process: rank 9--------
./32_gpu_rank_9_bs64_node64_resnet152.log-Num devices: 1
./32_gpu_rank_9_bs64_node64_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_9_bs64_node64_resnet152.log-Time per mini-batch : 0.6949597716331481
./32_gpu_rank_9_bs64_node64_resnet152.log-Throughput [img/sec] : 92.09166143473409
./32_gpu_rank_9_bs64_node64_resnet152.log-
./32_gpu_rank_9_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_9_bs64_node64_resnet152.log-Num devices: 32
./32_gpu_rank_9_bs64_node64_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_9_bs64_node64_resnet152.log-Time per mini-batch : 0.6949597716331481
./32_gpu_rank_9_bs64_node64_resnet152.log-Throughput [img/sec] : 2946.933165911491
./4_gpu_rank_1_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./4_gpu_rank_1_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./4_gpu_rank_1_bs64_node64_resnet152.log---------This process: rank 1--------
./4_gpu_rank_1_bs64_node64_resnet152.log-Num devices: 1
./4_gpu_rank_1_bs64_node64_resnet152.log-Mini batch size [img] : 64
./4_gpu_rank_1_bs64_node64_resnet152.log-Time per mini-batch : 0.7009250712394715
./4_gpu_rank_1_bs64_node64_resnet152.log-Throughput [img/sec] : 91.30790526129485
./4_gpu_rank_1_bs64_node64_resnet152.log-
./4_gpu_rank_1_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_1_bs64_node64_resnet152.log-Num devices: 4
./4_gpu_rank_1_bs64_node64_resnet152.log-Mini batch size [img] : 256
./4_gpu_rank_1_bs64_node64_resnet152.log-Time per mini-batch : 0.7009250712394715
./4_gpu_rank_1_bs64_node64_resnet152.log-Throughput [img/sec] : 365.2316210451794
./32_gpu_rank_14_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_14_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_14_bs64_node64_resnet101.log---------This process: rank 14--------
./32_gpu_rank_14_bs64_node64_resnet101.log-Num devices: 1
./32_gpu_rank_14_bs64_node64_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_14_bs64_node64_resnet101.log-Time per mini-batch : 0.49063706636428833
./32_gpu_rank_14_bs64_node64_resnet101.log-Throughput [img/sec] : 130.44265178383662
./32_gpu_rank_14_bs64_node64_resnet101.log-
./32_gpu_rank_14_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_14_bs64_node64_resnet101.log-Num devices: 32
./32_gpu_rank_14_bs64_node64_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_14_bs64_node64_resnet101.log-Time per mini-batch : 0.49063706636428833
./32_gpu_rank_14_bs64_node64_resnet101.log-Throughput [img/sec] : 4174.164857082772
./16_gpu_rank_7_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_7_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_7_bs64_node64_resnet101.log---------This process: rank 7--------
./16_gpu_rank_7_bs64_node64_resnet101.log-Num devices: 1
./16_gpu_rank_7_bs64_node64_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_7_bs64_node64_resnet101.log-Time per mini-batch : 0.49634103059768675
./16_gpu_rank_7_bs64_node64_resnet101.log-Throughput [img/sec] : 128.94360138417755
./16_gpu_rank_7_bs64_node64_resnet101.log-
./16_gpu_rank_7_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_7_bs64_node64_resnet101.log-Num devices: 16
./16_gpu_rank_7_bs64_node64_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_7_bs64_node64_resnet101.log-Time per mini-batch : 0.49634103059768675
./16_gpu_rank_7_bs64_node64_resnet101.log-Throughput [img/sec] : 2063.097622146841
./8_gpu_rank_3_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./8_gpu_rank_3_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./8_gpu_rank_3_bs64_node64_resnet101.log---------This process: rank 3--------
./8_gpu_rank_3_bs64_node64_resnet101.log-Num devices: 1
./8_gpu_rank_3_bs64_node64_resnet101.log-Mini batch size [img] : 64
./8_gpu_rank_3_bs64_node64_resnet101.log-Time per mini-batch : 0.4817300248146057
./8_gpu_rank_3_bs64_node64_resnet101.log-Throughput [img/sec] : 132.85449671655917
./8_gpu_rank_3_bs64_node64_resnet101.log-
./8_gpu_rank_3_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_3_bs64_node64_resnet101.log-Num devices: 8
./8_gpu_rank_3_bs64_node64_resnet101.log-Mini batch size [img] : 512
./8_gpu_rank_3_bs64_node64_resnet101.log-Time per mini-batch : 0.4817300248146057
./8_gpu_rank_3_bs64_node64_resnet101.log-Throughput [img/sec] : 1062.8359737324733
./8_gpu_rank_3_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./8_gpu_rank_3_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./8_gpu_rank_3_bs64_node64_resnet50.log---------This process: rank 3--------
./8_gpu_rank_3_bs64_node64_resnet50.log-Num devices: 1
./8_gpu_rank_3_bs64_node64_resnet50.log-Mini batch size [img] : 64
./8_gpu_rank_3_bs64_node64_resnet50.log-Time per mini-batch : 0.2864000201225281
./8_gpu_rank_3_bs64_node64_resnet50.log-Throughput [img/sec] : 223.46367145023044
./8_gpu_rank_3_bs64_node64_resnet50.log-
./8_gpu_rank_3_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_3_bs64_node64_resnet50.log-Num devices: 8
./8_gpu_rank_3_bs64_node64_resnet50.log-Mini batch size [img] : 512
./8_gpu_rank_3_bs64_node64_resnet50.log-Time per mini-batch : 0.2864000201225281
./8_gpu_rank_3_bs64_node64_resnet50.log-Throughput [img/sec] : 1787.7093716018435
./32_gpu_rank_15_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_15_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_15_bs64_node64_resnet152.log---------This process: rank 15--------
./32_gpu_rank_15_bs64_node64_resnet152.log-Num devices: 1
./32_gpu_rank_15_bs64_node64_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_15_bs64_node64_resnet152.log-Time per mini-batch : 0.6959027767181396
./32_gpu_rank_15_bs64_node64_resnet152.log-Throughput [img/sec] : 91.96686971393106
./32_gpu_rank_15_bs64_node64_resnet152.log-
./32_gpu_rank_15_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_15_bs64_node64_resnet152.log-Num devices: 32
./32_gpu_rank_15_bs64_node64_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_15_bs64_node64_resnet152.log-Time per mini-batch : 0.6959027767181396
./32_gpu_rank_15_bs64_node64_resnet152.log-Throughput [img/sec] : 2942.939830845794
./32_gpu_rank_8_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_8_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_8_bs64_node64_resnet152.log---------This process: rank 8--------
./32_gpu_rank_8_bs64_node64_resnet152.log-Num devices: 1
./32_gpu_rank_8_bs64_node64_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_8_bs64_node64_resnet152.log-Time per mini-batch : 0.6947743630409241
./32_gpu_rank_8_bs64_node64_resnet152.log-Throughput [img/sec] : 92.11623716206441
./32_gpu_rank_8_bs64_node64_resnet152.log-
./32_gpu_rank_8_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_8_bs64_node64_resnet152.log-Num devices: 32
./32_gpu_rank_8_bs64_node64_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_8_bs64_node64_resnet152.log-Time per mini-batch : 0.6947743630409241
./32_gpu_rank_8_bs64_node64_resnet152.log-Throughput [img/sec] : 2947.719589186061
./32_gpu_rank_8_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_8_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_8_bs64_node64_resnet101.log---------This process: rank 8--------
./32_gpu_rank_8_bs64_node64_resnet101.log-Num devices: 1
./32_gpu_rank_8_bs64_node64_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_8_bs64_node64_resnet101.log-Time per mini-batch : 0.48953667402267453
./32_gpu_rank_8_bs64_node64_resnet101.log-Throughput [img/sec] : 130.73586392229242
./32_gpu_rank_8_bs64_node64_resnet101.log-
./32_gpu_rank_8_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_8_bs64_node64_resnet101.log-Num devices: 32
./32_gpu_rank_8_bs64_node64_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_8_bs64_node64_resnet101.log-Time per mini-batch : 0.48953667402267453
./32_gpu_rank_8_bs64_node64_resnet101.log-Throughput [img/sec] : 4183.547645513358
./16_gpu_rank_4_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_4_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_4_bs64_node64_resnet101.log---------This process: rank 4--------
./16_gpu_rank_4_bs64_node64_resnet101.log-Num devices: 1
./16_gpu_rank_4_bs64_node64_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_4_bs64_node64_resnet101.log-Time per mini-batch : 0.4956852173805237
./16_gpu_rank_4_bs64_node64_resnet101.log-Throughput [img/sec] : 129.11419940705835
./16_gpu_rank_4_bs64_node64_resnet101.log-
./16_gpu_rank_4_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_4_bs64_node64_resnet101.log-Num devices: 16
./16_gpu_rank_4_bs64_node64_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_4_bs64_node64_resnet101.log-Time per mini-batch : 0.4956852173805237
./16_gpu_rank_4_bs64_node64_resnet101.log-Throughput [img/sec] : 2065.8271905129336
./32_gpu_rank_11_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_11_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_11_bs64_node64_resnet152.log---------This process: rank 11--------
./32_gpu_rank_11_bs64_node64_resnet152.log-Num devices: 1
./32_gpu_rank_11_bs64_node64_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_11_bs64_node64_resnet152.log-Time per mini-batch : 0.6946861577033997
./32_gpu_rank_11_bs64_node64_resnet152.log-Throughput [img/sec] : 92.12793329808245
./32_gpu_rank_11_bs64_node64_resnet152.log-
./32_gpu_rank_11_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_11_bs64_node64_resnet152.log-Num devices: 32
./32_gpu_rank_11_bs64_node64_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_11_bs64_node64_resnet152.log-Time per mini-batch : 0.6946861577033997
./32_gpu_rank_11_bs64_node64_resnet152.log-Throughput [img/sec] : 2948.0938655386385
./16_gpu_rank_4_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_4_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_4_bs64_node64_resnet152.log---------This process: rank 4--------
./16_gpu_rank_4_bs64_node64_resnet152.log-Num devices: 1
./16_gpu_rank_4_bs64_node64_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_4_bs64_node64_resnet152.log-Time per mini-batch : 0.7007318544387817
./16_gpu_rank_4_bs64_node64_resnet152.log-Throughput [img/sec] : 91.33308211206952
./16_gpu_rank_4_bs64_node64_resnet152.log-
./16_gpu_rank_4_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_4_bs64_node64_resnet152.log-Num devices: 16
./16_gpu_rank_4_bs64_node64_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_4_bs64_node64_resnet152.log-Time per mini-batch : 0.7007318544387817
./16_gpu_rank_4_bs64_node64_resnet152.log-Throughput [img/sec] : 1461.3293137931123
./32_gpu_rank_9_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_9_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_9_bs64_node64_resnet101.log---------This process: rank 9--------
./32_gpu_rank_9_bs64_node64_resnet101.log-Num devices: 1
./32_gpu_rank_9_bs64_node64_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_9_bs64_node64_resnet101.log-Time per mini-batch : 0.4890637588500977
./32_gpu_rank_9_bs64_node64_resnet101.log-Throughput [img/sec] : 130.86228296792802
./32_gpu_rank_9_bs64_node64_resnet101.log-
./32_gpu_rank_9_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_9_bs64_node64_resnet101.log-Num devices: 32
./32_gpu_rank_9_bs64_node64_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_9_bs64_node64_resnet101.log-Time per mini-batch : 0.4890637588500977
./32_gpu_rank_9_bs64_node64_resnet101.log-Throughput [img/sec] : 4187.593054973697
./16_gpu_rank_5_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_5_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_5_bs64_node64_resnet101.log---------This process: rank 5--------
./16_gpu_rank_5_bs64_node64_resnet101.log-Num devices: 1
./16_gpu_rank_5_bs64_node64_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_5_bs64_node64_resnet101.log-Time per mini-batch : 0.4975856566429138
./16_gpu_rank_5_bs64_node64_resnet101.log-Throughput [img/sec] : 128.62107085600502
./16_gpu_rank_5_bs64_node64_resnet101.log-
./16_gpu_rank_5_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_5_bs64_node64_resnet101.log-Num devices: 16
./16_gpu_rank_5_bs64_node64_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_5_bs64_node64_resnet101.log-Time per mini-batch : 0.4975856566429138
./16_gpu_rank_5_bs64_node64_resnet101.log-Throughput [img/sec] : 2057.9371336960803
./16_gpu_rank_6_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_6_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_6_bs64_node64_resnet152.log---------This process: rank 6--------
./16_gpu_rank_6_bs64_node64_resnet152.log-Num devices: 1
./16_gpu_rank_6_bs64_node64_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_6_bs64_node64_resnet152.log-Time per mini-batch : 0.7009049248695374
./16_gpu_rank_6_bs64_node64_resnet152.log-Throughput [img/sec] : 91.31052975824447
./16_gpu_rank_6_bs64_node64_resnet152.log-
./16_gpu_rank_6_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_6_bs64_node64_resnet152.log-Num devices: 16
./16_gpu_rank_6_bs64_node64_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_6_bs64_node64_resnet152.log-Time per mini-batch : 0.7009049248695374
./16_gpu_rank_6_bs64_node64_resnet152.log-Throughput [img/sec] : 1460.9684761319115
./16_gpu_rank_7_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_7_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_7_bs64_node64_resnet152.log---------This process: rank 7--------
./16_gpu_rank_7_bs64_node64_resnet152.log-Num devices: 1
./16_gpu_rank_7_bs64_node64_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_7_bs64_node64_resnet152.log-Time per mini-batch : 0.6988066458702087
./16_gpu_rank_7_bs64_node64_resnet152.log-Throughput [img/sec] : 91.58470426437086
./16_gpu_rank_7_bs64_node64_resnet152.log-
./16_gpu_rank_7_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_7_bs64_node64_resnet152.log-Num devices: 16
./16_gpu_rank_7_bs64_node64_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_7_bs64_node64_resnet152.log-Time per mini-batch : 0.6988066458702087
./16_gpu_rank_7_bs64_node64_resnet152.log-Throughput [img/sec] : 1465.3552682299337
./32_gpu_rank_12_bs64_node64_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_12_bs64_node64_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_12_bs64_node64_resnet152.log---------This process: rank 12--------
./32_gpu_rank_12_bs64_node64_resnet152.log-Num devices: 1
./32_gpu_rank_12_bs64_node64_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_12_bs64_node64_resnet152.log-Time per mini-batch : 0.6914739751815796
./32_gpu_rank_12_bs64_node64_resnet152.log-Throughput [img/sec] : 92.55590564083592
./32_gpu_rank_12_bs64_node64_resnet152.log-
./32_gpu_rank_12_bs64_node64_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_12_bs64_node64_resnet152.log-Num devices: 32
./32_gpu_rank_12_bs64_node64_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_12_bs64_node64_resnet152.log-Time per mini-batch : 0.6914739751815796
./32_gpu_rank_12_bs64_node64_resnet152.log-Throughput [img/sec] : 2961.7889805067493
./8_gpu_rank_2_bs64_node64_resnet101.log:--------------------SUMMARY--------------------------
./8_gpu_rank_2_bs64_node64_resnet101.log-Microbenchmark for network : resnet101
./8_gpu_rank_2_bs64_node64_resnet101.log---------This process: rank 2--------
./8_gpu_rank_2_bs64_node64_resnet101.log-Num devices: 1
./8_gpu_rank_2_bs64_node64_resnet101.log-Mini batch size [img] : 64
./8_gpu_rank_2_bs64_node64_resnet101.log-Time per mini-batch : 0.48081016302108764
./8_gpu_rank_2_bs64_node64_resnet101.log-Throughput [img/sec] : 133.1086672500162
./8_gpu_rank_2_bs64_node64_resnet101.log-
./8_gpu_rank_2_bs64_node64_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_2_bs64_node64_resnet101.log-Num devices: 8
./8_gpu_rank_2_bs64_node64_resnet101.log-Mini batch size [img] : 512
./8_gpu_rank_2_bs64_node64_resnet101.log-Time per mini-batch : 0.48081016302108764
./8_gpu_rank_2_bs64_node64_resnet101.log-Throughput [img/sec] : 1064.8693380001296
./32_gpu_rank_10_bs64_node64_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_10_bs64_node64_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_10_bs64_node64_resnet50.log---------This process: rank 10--------
./32_gpu_rank_10_bs64_node64_resnet50.log-Num devices: 1
./32_gpu_rank_10_bs64_node64_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_10_bs64_node64_resnet50.log-Time per mini-batch : 0.2913209867477417
./32_gpu_rank_10_bs64_node64_resnet50.log-Throughput [img/sec] : 219.6889441934314
./32_gpu_rank_10_bs64_node64_resnet50.log-
./32_gpu_rank_10_bs64_node64_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_10_bs64_node64_resnet50.log-Num devices: 32
./32_gpu_rank_10_bs64_node64_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_10_bs64_node64_resnet50.log-Time per mini-batch : 0.2913209867477417
./32_gpu_rank_10_bs64_node64_resnet50.log-Throughput [img/sec] : 7030.046214189805

  ======
    
  
  ./4_gpu_rank_2_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./4_gpu_rank_2_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./4_gpu_rank_2_bs64_node65_resnet50.log---------This process: rank 2--------
./4_gpu_rank_2_bs64_node65_resnet50.log-Num devices: 1
./4_gpu_rank_2_bs64_node65_resnet50.log-Mini batch size [img] : 64
./4_gpu_rank_2_bs64_node65_resnet50.log-Time per mini-batch : 0.2822937250137329
./4_gpu_rank_2_bs64_node65_resnet50.log-Throughput [img/sec] : 226.71421405802258
./4_gpu_rank_2_bs64_node65_resnet50.log-
./4_gpu_rank_2_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_2_bs64_node65_resnet50.log-Num devices: 4
./4_gpu_rank_2_bs64_node65_resnet50.log-Mini batch size [img] : 256
./4_gpu_rank_2_bs64_node65_resnet50.log-Time per mini-batch : 0.2822937250137329
./4_gpu_rank_2_bs64_node65_resnet50.log-Throughput [img/sec] : 906.8568562320903
./8_gpu_rank_4_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./8_gpu_rank_4_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./8_gpu_rank_4_bs64_node65_resnet50.log---------This process: rank 4--------
./8_gpu_rank_4_bs64_node65_resnet50.log-Num devices: 1
./8_gpu_rank_4_bs64_node65_resnet50.log-Mini batch size [img] : 64
./8_gpu_rank_4_bs64_node65_resnet50.log-Time per mini-batch : 0.28528759479522703
./8_gpu_rank_4_bs64_node65_resnet50.log-Throughput [img/sec] : 224.33502601449513
./8_gpu_rank_4_bs64_node65_resnet50.log-
./8_gpu_rank_4_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_4_bs64_node65_resnet50.log-Num devices: 8
./8_gpu_rank_4_bs64_node65_resnet50.log-Mini batch size [img] : 512
./8_gpu_rank_4_bs64_node65_resnet50.log-Time per mini-batch : 0.28528759479522703
./8_gpu_rank_4_bs64_node65_resnet50.log-Throughput [img/sec] : 1794.680208115961
./8_gpu_rank_5_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./8_gpu_rank_5_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./8_gpu_rank_5_bs64_node65_resnet50.log---------This process: rank 5--------
./8_gpu_rank_5_bs64_node65_resnet50.log-Num devices: 1
./8_gpu_rank_5_bs64_node65_resnet50.log-Mini batch size [img] : 64
./8_gpu_rank_5_bs64_node65_resnet50.log-Time per mini-batch : 0.2844350719451904
./8_gpu_rank_5_bs64_node65_resnet50.log-Throughput [img/sec] : 225.0074140376492
./8_gpu_rank_5_bs64_node65_resnet50.log-
./8_gpu_rank_5_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_5_bs64_node65_resnet50.log-Num devices: 8
./8_gpu_rank_5_bs64_node65_resnet50.log-Mini batch size [img] : 512
./8_gpu_rank_5_bs64_node65_resnet50.log-Time per mini-batch : 0.2844350719451904
./8_gpu_rank_5_bs64_node65_resnet50.log-Throughput [img/sec] : 1800.0593123011936
./16_gpu_rank_8_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_8_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_8_bs64_node65_resnet50.log---------This process: rank 8--------
./16_gpu_rank_8_bs64_node65_resnet50.log-Num devices: 1
./16_gpu_rank_8_bs64_node65_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_8_bs64_node65_resnet50.log-Time per mini-batch : 0.2877618741989136
./16_gpu_rank_8_bs64_node65_resnet50.log-Throughput [img/sec] : 222.40611331214922
./16_gpu_rank_8_bs64_node65_resnet50.log-
./16_gpu_rank_8_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_8_bs64_node65_resnet50.log-Num devices: 16
./16_gpu_rank_8_bs64_node65_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_8_bs64_node65_resnet50.log-Time per mini-batch : 0.2877618741989136
./16_gpu_rank_8_bs64_node65_resnet50.log-Throughput [img/sec] : 3558.4978129943875
./16_gpu_rank_9_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_9_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_9_bs64_node65_resnet50.log---------This process: rank 9--------
./16_gpu_rank_9_bs64_node65_resnet50.log-Num devices: 1
./16_gpu_rank_9_bs64_node65_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_9_bs64_node65_resnet50.log-Time per mini-batch : 0.2867882299423218
./16_gpu_rank_9_bs64_node65_resnet50.log-Throughput [img/sec] : 223.16118068329212
./16_gpu_rank_9_bs64_node65_resnet50.log-
./16_gpu_rank_9_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_9_bs64_node65_resnet50.log-Num devices: 16
./16_gpu_rank_9_bs64_node65_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_9_bs64_node65_resnet50.log-Time per mini-batch : 0.2867882299423218
./16_gpu_rank_9_bs64_node65_resnet50.log-Throughput [img/sec] : 3570.578890932674
./16_gpu_rank_10_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_10_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_10_bs64_node65_resnet50.log---------This process: rank 10--------
./16_gpu_rank_10_bs64_node65_resnet50.log-Num devices: 1
./16_gpu_rank_10_bs64_node65_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_10_bs64_node65_resnet50.log-Time per mini-batch : 0.2878041052818298
./16_gpu_rank_10_bs64_node65_resnet50.log-Throughput [img/sec] : 222.3734784371075
./16_gpu_rank_10_bs64_node65_resnet50.log-
./16_gpu_rank_10_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_10_bs64_node65_resnet50.log-Num devices: 16
./16_gpu_rank_10_bs64_node65_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_10_bs64_node65_resnet50.log-Time per mini-batch : 0.2878041052818298
./16_gpu_rank_10_bs64_node65_resnet50.log-Throughput [img/sec] : 3557.97565499372
./16_gpu_rank_11_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./16_gpu_rank_11_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./16_gpu_rank_11_bs64_node65_resnet50.log---------This process: rank 11--------
./16_gpu_rank_11_bs64_node65_resnet50.log-Num devices: 1
./16_gpu_rank_11_bs64_node65_resnet50.log-Mini batch size [img] : 64
./16_gpu_rank_11_bs64_node65_resnet50.log-Time per mini-batch : 0.2870940685272217
./16_gpu_rank_11_bs64_node65_resnet50.log-Throughput [img/sec] : 222.92344919669299
./16_gpu_rank_11_bs64_node65_resnet50.log-
./16_gpu_rank_11_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_11_bs64_node65_resnet50.log-Num devices: 16
./16_gpu_rank_11_bs64_node65_resnet50.log-Mini batch size [img] : 1024
./16_gpu_rank_11_bs64_node65_resnet50.log-Time per mini-batch : 0.2870940685272217
./16_gpu_rank_11_bs64_node65_resnet50.log-Throughput [img/sec] : 3566.7751871470878
./32_gpu_rank_16_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_16_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_16_bs64_node65_resnet50.log---------This process: rank 16--------
./32_gpu_rank_16_bs64_node65_resnet50.log-Num devices: 1
./32_gpu_rank_16_bs64_node65_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_16_bs64_node65_resnet50.log-Time per mini-batch : 0.2922119140625
./32_gpu_rank_16_bs64_node65_resnet50.log-Throughput [img/sec] : 219.01913275962903
./32_gpu_rank_16_bs64_node65_resnet50.log-
./32_gpu_rank_16_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_16_bs64_node65_resnet50.log-Num devices: 32
./32_gpu_rank_16_bs64_node65_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_16_bs64_node65_resnet50.log-Time per mini-batch : 0.2922119140625
./32_gpu_rank_16_bs64_node65_resnet50.log-Throughput [img/sec] : 7008.612248308129
./32_gpu_rank_17_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_17_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_17_bs64_node65_resnet50.log---------This process: rank 17--------
./32_gpu_rank_17_bs64_node65_resnet50.log-Num devices: 1
./32_gpu_rank_17_bs64_node65_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_17_bs64_node65_resnet50.log-Time per mini-batch : 0.29013142347335813
./32_gpu_rank_17_bs64_node65_resnet50.log-Throughput [img/sec] : 220.5896873693067
./32_gpu_rank_17_bs64_node65_resnet50.log-
./32_gpu_rank_17_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_17_bs64_node65_resnet50.log-Num devices: 32
./32_gpu_rank_17_bs64_node65_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_17_bs64_node65_resnet50.log-Time per mini-batch : 0.29013142347335813
./32_gpu_rank_17_bs64_node65_resnet50.log-Throughput [img/sec] : 7058.869995817814
./32_gpu_rank_18_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_18_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_18_bs64_node65_resnet50.log---------This process: rank 18--------
./32_gpu_rank_18_bs64_node65_resnet50.log-Num devices: 1
./32_gpu_rank_18_bs64_node65_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_18_bs64_node65_resnet50.log-Time per mini-batch : 0.29145357370376584
./32_gpu_rank_18_bs64_node65_resnet50.log-Throughput [img/sec] : 219.5890041308938
./32_gpu_rank_18_bs64_node65_resnet50.log-
./32_gpu_rank_18_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_18_bs64_node65_resnet50.log-Num devices: 32
./32_gpu_rank_18_bs64_node65_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_18_bs64_node65_resnet50.log-Time per mini-batch : 0.29145357370376584
./32_gpu_rank_18_bs64_node65_resnet50.log-Throughput [img/sec] : 7026.848132188602
./32_gpu_rank_19_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_19_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_19_bs64_node65_resnet50.log---------This process: rank 19--------
./32_gpu_rank_19_bs64_node65_resnet50.log-Num devices: 1
./32_gpu_rank_19_bs64_node65_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_19_bs64_node65_resnet50.log-Time per mini-batch : 0.28996479988098145
./32_gpu_rank_19_bs64_node65_resnet50.log-Throughput [img/sec] : 220.7164456729553
./32_gpu_rank_19_bs64_node65_resnet50.log-
./32_gpu_rank_19_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_19_bs64_node65_resnet50.log-Num devices: 32
./32_gpu_rank_19_bs64_node65_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_19_bs64_node65_resnet50.log-Time per mini-batch : 0.28996479988098145
./32_gpu_rank_19_bs64_node65_resnet50.log-Throughput [img/sec] : 7062.92626153457
./32_gpu_rank_20_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_20_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_20_bs64_node65_resnet50.log---------This process: rank 20--------
./32_gpu_rank_20_bs64_node65_resnet50.log-Num devices: 1
./32_gpu_rank_20_bs64_node65_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_20_bs64_node65_resnet50.log-Time per mini-batch : 0.2887453746795654
./32_gpu_rank_20_bs64_node65_resnet50.log-Throughput [img/sec] : 221.64857210621597
./32_gpu_rank_20_bs64_node65_resnet50.log-
./32_gpu_rank_20_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_20_bs64_node65_resnet50.log-Num devices: 32
./32_gpu_rank_20_bs64_node65_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_20_bs64_node65_resnet50.log-Time per mini-batch : 0.2887453746795654
./32_gpu_rank_20_bs64_node65_resnet50.log-Throughput [img/sec] : 7092.754307398911
./32_gpu_rank_21_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_21_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_21_bs64_node65_resnet50.log---------This process: rank 21--------
./32_gpu_rank_21_bs64_node65_resnet50.log-Num devices: 1
./32_gpu_rank_21_bs64_node65_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_21_bs64_node65_resnet50.log-Time per mini-batch : 0.29218133449554445
./32_gpu_rank_21_bs64_node65_resnet50.log-Throughput [img/sec] : 219.0420552034817
./32_gpu_rank_21_bs64_node65_resnet50.log-
./32_gpu_rank_21_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_21_bs64_node65_resnet50.log-Num devices: 32
./32_gpu_rank_21_bs64_node65_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_21_bs64_node65_resnet50.log-Time per mini-batch : 0.29218133449554445
./32_gpu_rank_21_bs64_node65_resnet50.log-Throughput [img/sec] : 7009.345766511415
./32_gpu_rank_22_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_22_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_22_bs64_node65_resnet50.log---------This process: rank 22--------
./32_gpu_rank_22_bs64_node65_resnet50.log-Num devices: 1
./32_gpu_rank_22_bs64_node65_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_22_bs64_node65_resnet50.log-Time per mini-batch : 0.29087244272232055
./32_gpu_rank_22_bs64_node65_resnet50.log-Throughput [img/sec] : 220.02771868319329
./32_gpu_rank_22_bs64_node65_resnet50.log-
./32_gpu_rank_22_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_22_bs64_node65_resnet50.log-Num devices: 32
./32_gpu_rank_22_bs64_node65_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_22_bs64_node65_resnet50.log-Time per mini-batch : 0.29087244272232055
./32_gpu_rank_22_bs64_node65_resnet50.log-Throughput [img/sec] : 7040.886997862185
./32_gpu_rank_23_bs64_node65_resnet50.log:--------------------SUMMARY--------------------------
./32_gpu_rank_23_bs64_node65_resnet50.log-Microbenchmark for network : resnet50
./32_gpu_rank_23_bs64_node65_resnet50.log---------This process: rank 23--------
./32_gpu_rank_23_bs64_node65_resnet50.log-Num devices: 1
./32_gpu_rank_23_bs64_node65_resnet50.log-Mini batch size [img] : 64
./32_gpu_rank_23_bs64_node65_resnet50.log-Time per mini-batch : 0.2930854272842407
./32_gpu_rank_23_bs64_node65_resnet50.log-Throughput [img/sec] : 218.3663670794911
./32_gpu_rank_23_bs64_node65_resnet50.log-
./32_gpu_rank_23_bs64_node65_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_23_bs64_node65_resnet50.log-Num devices: 32
./32_gpu_rank_23_bs64_node65_resnet50.log-Mini batch size [img] : 2048
./32_gpu_rank_23_bs64_node65_resnet50.log-Time per mini-batch : 0.2930854272842407
./32_gpu_rank_23_bs64_node65_resnet50.log-Throughput [img/sec] : 6987.723746543716
./4_gpu_rank_2_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./4_gpu_rank_2_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./4_gpu_rank_2_bs64_node65_resnet101.log---------This process: rank 2--------
./4_gpu_rank_2_bs64_node65_resnet101.log-Num devices: 1
./4_gpu_rank_2_bs64_node65_resnet101.log-Mini batch size [img] : 64
./4_gpu_rank_2_bs64_node65_resnet101.log-Time per mini-batch : 0.4918133020401001
./4_gpu_rank_2_bs64_node65_resnet101.log-Throughput [img/sec] : 130.1306811640116
./4_gpu_rank_2_bs64_node65_resnet101.log-
./4_gpu_rank_2_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_2_bs64_node65_resnet101.log-Num devices: 4
./4_gpu_rank_2_bs64_node65_resnet101.log-Mini batch size [img] : 256
./4_gpu_rank_2_bs64_node65_resnet101.log-Time per mini-batch : 0.4918133020401001
./4_gpu_rank_2_bs64_node65_resnet101.log-Throughput [img/sec] : 520.5227246560464
./8_gpu_rank_4_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./8_gpu_rank_4_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./8_gpu_rank_4_bs64_node65_resnet101.log---------This process: rank 4--------
./8_gpu_rank_4_bs64_node65_resnet101.log-Num devices: 1
./8_gpu_rank_4_bs64_node65_resnet101.log-Mini batch size [img] : 64
./8_gpu_rank_4_bs64_node65_resnet101.log-Time per mini-batch : 0.4819255256652832
./8_gpu_rank_4_bs64_node65_resnet101.log-Throughput [img/sec] : 132.80060215040487
./8_gpu_rank_4_bs64_node65_resnet101.log-
./8_gpu_rank_4_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_4_bs64_node65_resnet101.log-Num devices: 8
./8_gpu_rank_4_bs64_node65_resnet101.log-Mini batch size [img] : 512
./8_gpu_rank_4_bs64_node65_resnet101.log-Time per mini-batch : 0.4819255256652832
./8_gpu_rank_4_bs64_node65_resnet101.log-Throughput [img/sec] : 1062.404817203239
./8_gpu_rank_5_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./8_gpu_rank_5_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./8_gpu_rank_5_bs64_node65_resnet101.log---------This process: rank 5--------
./8_gpu_rank_5_bs64_node65_resnet101.log-Num devices: 1
./8_gpu_rank_5_bs64_node65_resnet101.log-Mini batch size [img] : 64
./8_gpu_rank_5_bs64_node65_resnet101.log-Time per mini-batch : 0.4785850787162781
./8_gpu_rank_5_bs64_node65_resnet101.log-Throughput [img/sec] : 133.72752901462988
./8_gpu_rank_5_bs64_node65_resnet101.log-
./8_gpu_rank_5_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_5_bs64_node65_resnet101.log-Num devices: 8
./8_gpu_rank_5_bs64_node65_resnet101.log-Mini batch size [img] : 512
./8_gpu_rank_5_bs64_node65_resnet101.log-Time per mini-batch : 0.4785850787162781
./8_gpu_rank_5_bs64_node65_resnet101.log-Throughput [img/sec] : 1069.820232117039
./16_gpu_rank_8_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_8_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_8_bs64_node65_resnet101.log---------This process: rank 8--------
./16_gpu_rank_8_bs64_node65_resnet101.log-Num devices: 1
./16_gpu_rank_8_bs64_node65_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_8_bs64_node65_resnet101.log-Time per mini-batch : 0.4963167071342468
./16_gpu_rank_8_bs64_node65_resnet101.log-Throughput [img/sec] : 128.94992064550607
./16_gpu_rank_8_bs64_node65_resnet101.log-
./16_gpu_rank_8_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_8_bs64_node65_resnet101.log-Num devices: 16
./16_gpu_rank_8_bs64_node65_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_8_bs64_node65_resnet101.log-Time per mini-batch : 0.4963167071342468
./16_gpu_rank_8_bs64_node65_resnet101.log-Throughput [img/sec] : 2063.198730328097
./16_gpu_rank_9_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_9_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_9_bs64_node65_resnet101.log---------This process: rank 9--------
./16_gpu_rank_9_bs64_node65_resnet101.log-Num devices: 1
./16_gpu_rank_9_bs64_node65_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_9_bs64_node65_resnet101.log-Time per mini-batch : 0.4933263182640076
./16_gpu_rank_9_bs64_node65_resnet101.log-Throughput [img/sec] : 129.73157447835547
./16_gpu_rank_9_bs64_node65_resnet101.log-
./16_gpu_rank_9_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_9_bs64_node65_resnet101.log-Num devices: 16
./16_gpu_rank_9_bs64_node65_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_9_bs64_node65_resnet101.log-Time per mini-batch : 0.4933263182640076
./16_gpu_rank_9_bs64_node65_resnet101.log-Throughput [img/sec] : 2075.7051916536875
./16_gpu_rank_10_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_10_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_10_bs64_node65_resnet101.log---------This process: rank 10--------
./16_gpu_rank_10_bs64_node65_resnet101.log-Num devices: 1
./16_gpu_rank_10_bs64_node65_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_10_bs64_node65_resnet101.log-Time per mini-batch : 0.4975338673591614
./16_gpu_rank_10_bs64_node65_resnet101.log-Throughput [img/sec] : 128.63445927752184
./16_gpu_rank_10_bs64_node65_resnet101.log-
./16_gpu_rank_10_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_10_bs64_node65_resnet101.log-Num devices: 16
./16_gpu_rank_10_bs64_node65_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_10_bs64_node65_resnet101.log-Time per mini-batch : 0.4975338673591614
./16_gpu_rank_10_bs64_node65_resnet101.log-Throughput [img/sec] : 2058.1513484403495
./16_gpu_rank_11_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./16_gpu_rank_11_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./16_gpu_rank_11_bs64_node65_resnet101.log---------This process: rank 11--------
./16_gpu_rank_11_bs64_node65_resnet101.log-Num devices: 1
./16_gpu_rank_11_bs64_node65_resnet101.log-Mini batch size [img] : 64
./16_gpu_rank_11_bs64_node65_resnet101.log-Time per mini-batch : 0.4980280661582947
./16_gpu_rank_11_bs64_node65_resnet101.log-Throughput [img/sec] : 128.50681387032122
./16_gpu_rank_11_bs64_node65_resnet101.log-
./16_gpu_rank_11_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_11_bs64_node65_resnet101.log-Num devices: 16
./16_gpu_rank_11_bs64_node65_resnet101.log-Mini batch size [img] : 1024
./16_gpu_rank_11_bs64_node65_resnet101.log-Time per mini-batch : 0.4980280661582947
./16_gpu_rank_11_bs64_node65_resnet101.log-Throughput [img/sec] : 2056.1090219251396
./32_gpu_rank_16_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_16_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_16_bs64_node65_resnet101.log---------This process: rank 16--------
./32_gpu_rank_16_bs64_node65_resnet101.log-Num devices: 1
./32_gpu_rank_16_bs64_node65_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_16_bs64_node65_resnet101.log-Time per mini-batch : 0.49297739744186403
./32_gpu_rank_16_bs64_node65_resnet101.log-Throughput [img/sec] : 129.82339622892633
./32_gpu_rank_16_bs64_node65_resnet101.log-
./32_gpu_rank_16_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_16_bs64_node65_resnet101.log-Num devices: 32
./32_gpu_rank_16_bs64_node65_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_16_bs64_node65_resnet101.log-Time per mini-batch : 0.49297739744186403
./32_gpu_rank_16_bs64_node65_resnet101.log-Throughput [img/sec] : 4154.3486793256425
./32_gpu_rank_17_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_17_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_17_bs64_node65_resnet101.log---------This process: rank 17--------
./32_gpu_rank_17_bs64_node65_resnet101.log-Num devices: 1
./32_gpu_rank_17_bs64_node65_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_17_bs64_node65_resnet101.log-Time per mini-batch : 0.4861318111419678
./32_gpu_rank_17_bs64_node65_resnet101.log-Throughput [img/sec] : 131.651536750204
./32_gpu_rank_17_bs64_node65_resnet101.log-
./32_gpu_rank_17_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_17_bs64_node65_resnet101.log-Num devices: 32
./32_gpu_rank_17_bs64_node65_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_17_bs64_node65_resnet101.log-Time per mini-batch : 0.4861318111419678
./32_gpu_rank_17_bs64_node65_resnet101.log-Throughput [img/sec] : 4212.849176006528
./32_gpu_rank_18_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_18_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_18_bs64_node65_resnet101.log---------This process: rank 18--------
./32_gpu_rank_18_bs64_node65_resnet101.log-Num devices: 1
./32_gpu_rank_18_bs64_node65_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_18_bs64_node65_resnet101.log-Time per mini-batch : 0.4931506252288818
./32_gpu_rank_18_bs64_node65_resnet101.log-Throughput [img/sec] : 129.77779348915197
./32_gpu_rank_18_bs64_node65_resnet101.log-
./32_gpu_rank_18_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_18_bs64_node65_resnet101.log-Num devices: 32
./32_gpu_rank_18_bs64_node65_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_18_bs64_node65_resnet101.log-Time per mini-batch : 0.4931506252288818
./32_gpu_rank_18_bs64_node65_resnet101.log-Throughput [img/sec] : 4152.889391652863
./32_gpu_rank_19_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_19_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_19_bs64_node65_resnet101.log---------This process: rank 19--------
./32_gpu_rank_19_bs64_node65_resnet101.log-Num devices: 1
./32_gpu_rank_19_bs64_node65_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_19_bs64_node65_resnet101.log-Time per mini-batch : 0.4933162760734558
./32_gpu_rank_19_bs64_node65_resnet101.log-Throughput [img/sec] : 129.73421535856698
./32_gpu_rank_19_bs64_node65_resnet101.log-
./32_gpu_rank_19_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_19_bs64_node65_resnet101.log-Num devices: 32
./32_gpu_rank_19_bs64_node65_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_19_bs64_node65_resnet101.log-Time per mini-batch : 0.4933162760734558
./32_gpu_rank_19_bs64_node65_resnet101.log-Throughput [img/sec] : 4151.4948914741435
./32_gpu_rank_20_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_20_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_20_bs64_node65_resnet101.log---------This process: rank 20--------
./32_gpu_rank_20_bs64_node65_resnet101.log-Num devices: 1
./32_gpu_rank_20_bs64_node65_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_20_bs64_node65_resnet101.log-Time per mini-batch : 0.49015216588974
./32_gpu_rank_20_bs64_node65_resnet101.log-Throughput [img/sec] : 130.57169681954815
./32_gpu_rank_20_bs64_node65_resnet101.log-
./32_gpu_rank_20_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_20_bs64_node65_resnet101.log-Num devices: 32
./32_gpu_rank_20_bs64_node65_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_20_bs64_node65_resnet101.log-Time per mini-batch : 0.49015216588974
./32_gpu_rank_20_bs64_node65_resnet101.log-Throughput [img/sec] : 4178.294298225541
./32_gpu_rank_21_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_21_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_21_bs64_node65_resnet101.log---------This process: rank 21--------
./32_gpu_rank_21_bs64_node65_resnet101.log-Num devices: 1
./32_gpu_rank_21_bs64_node65_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_21_bs64_node65_resnet101.log-Time per mini-batch : 0.4930442881584167
./32_gpu_rank_21_bs64_node65_resnet101.log-Throughput [img/sec] : 129.80578324727816
./32_gpu_rank_21_bs64_node65_resnet101.log-
./32_gpu_rank_21_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_21_bs64_node65_resnet101.log-Num devices: 32
./32_gpu_rank_21_bs64_node65_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_21_bs64_node65_resnet101.log-Time per mini-batch : 0.4930442881584167
./32_gpu_rank_21_bs64_node65_resnet101.log-Throughput [img/sec] : 4153.785063912901
./32_gpu_rank_22_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_22_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_22_bs64_node65_resnet101.log---------This process: rank 22--------
./32_gpu_rank_22_bs64_node65_resnet101.log-Num devices: 1
./32_gpu_rank_22_bs64_node65_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_22_bs64_node65_resnet101.log-Time per mini-batch : 0.49150933504104616
./32_gpu_rank_22_bs64_node65_resnet101.log-Throughput [img/sec] : 130.21115864392553
./32_gpu_rank_22_bs64_node65_resnet101.log-
./32_gpu_rank_22_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_22_bs64_node65_resnet101.log-Num devices: 32
./32_gpu_rank_22_bs64_node65_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_22_bs64_node65_resnet101.log-Time per mini-batch : 0.49150933504104616
./32_gpu_rank_22_bs64_node65_resnet101.log-Throughput [img/sec] : 4166.757076605617
./32_gpu_rank_23_bs64_node65_resnet101.log:--------------------SUMMARY--------------------------
./32_gpu_rank_23_bs64_node65_resnet101.log-Microbenchmark for network : resnet101
./32_gpu_rank_23_bs64_node65_resnet101.log---------This process: rank 23--------
./32_gpu_rank_23_bs64_node65_resnet101.log-Num devices: 1
./32_gpu_rank_23_bs64_node65_resnet101.log-Mini batch size [img] : 64
./32_gpu_rank_23_bs64_node65_resnet101.log-Time per mini-batch : 0.4898068642616272
./32_gpu_rank_23_bs64_node65_resnet101.log-Throughput [img/sec] : 130.66374661057182
./32_gpu_rank_23_bs64_node65_resnet101.log-
./32_gpu_rank_23_bs64_node65_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_23_bs64_node65_resnet101.log-Num devices: 32
./32_gpu_rank_23_bs64_node65_resnet101.log-Mini batch size [img] : 2048
./32_gpu_rank_23_bs64_node65_resnet101.log-Time per mini-batch : 0.4898068642616272
./32_gpu_rank_23_bs64_node65_resnet101.log-Throughput [img/sec] : 4181.239891538298
./4_gpu_rank_2_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./4_gpu_rank_2_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./4_gpu_rank_2_bs64_node65_resnet152.log---------This process: rank 2--------
./4_gpu_rank_2_bs64_node65_resnet152.log-Num devices: 1
./4_gpu_rank_2_bs64_node65_resnet152.log-Mini batch size [img] : 64
./4_gpu_rank_2_bs64_node65_resnet152.log-Time per mini-batch : 0.7002970695495605
./4_gpu_rank_2_bs64_node65_resnet152.log-Throughput [img/sec] : 91.38978696735883
./4_gpu_rank_2_bs64_node65_resnet152.log-
./4_gpu_rank_2_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./4_gpu_rank_2_bs64_node65_resnet152.log-Num devices: 4
./4_gpu_rank_2_bs64_node65_resnet152.log-Mini batch size [img] : 256
./4_gpu_rank_2_bs64_node65_resnet152.log-Time per mini-batch : 0.7002970695495605
./4_gpu_rank_2_bs64_node65_resnet152.log-Throughput [img/sec] : 365.5591478694353
./8_gpu_rank_4_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./8_gpu_rank_4_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./8_gpu_rank_4_bs64_node65_resnet152.log---------This process: rank 4--------
./8_gpu_rank_4_bs64_node65_resnet152.log-Num devices: 1
./8_gpu_rank_4_bs64_node65_resnet152.log-Mini batch size [img] : 64
./8_gpu_rank_4_bs64_node65_resnet152.log-Time per mini-batch : 0.6986356544494629
./8_gpu_rank_4_bs64_node65_resnet152.log-Throughput [img/sec] : 91.60711966587667
./8_gpu_rank_4_bs64_node65_resnet152.log-
./8_gpu_rank_4_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_4_bs64_node65_resnet152.log-Num devices: 8
./8_gpu_rank_4_bs64_node65_resnet152.log-Mini batch size [img] : 512
./8_gpu_rank_4_bs64_node65_resnet152.log-Time per mini-batch : 0.6986356544494629
./8_gpu_rank_4_bs64_node65_resnet152.log-Throughput [img/sec] : 732.8569573270133
./8_gpu_rank_5_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./8_gpu_rank_5_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./8_gpu_rank_5_bs64_node65_resnet152.log---------This process: rank 5--------
./8_gpu_rank_5_bs64_node65_resnet152.log-Num devices: 1
./8_gpu_rank_5_bs64_node65_resnet152.log-Mini batch size [img] : 64
./8_gpu_rank_5_bs64_node65_resnet152.log-Time per mini-batch : 0.6983098149299621
./8_gpu_rank_5_bs64_node65_resnet152.log-Throughput [img/sec] : 91.64986461835562
./8_gpu_rank_5_bs64_node65_resnet152.log-
./8_gpu_rank_5_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./8_gpu_rank_5_bs64_node65_resnet152.log-Num devices: 8
./8_gpu_rank_5_bs64_node65_resnet152.log-Mini batch size [img] : 512
./8_gpu_rank_5_bs64_node65_resnet152.log-Time per mini-batch : 0.6983098149299621
./8_gpu_rank_5_bs64_node65_resnet152.log-Throughput [img/sec] : 733.198916946845
./16_gpu_rank_8_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_8_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_8_bs64_node65_resnet152.log---------This process: rank 8--------
./16_gpu_rank_8_bs64_node65_resnet152.log-Num devices: 1
./16_gpu_rank_8_bs64_node65_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_8_bs64_node65_resnet152.log-Time per mini-batch : 0.6998129653930664
./16_gpu_rank_8_bs64_node65_resnet152.log-Throughput [img/sec] : 91.45300696744437
./16_gpu_rank_8_bs64_node65_resnet152.log-
./16_gpu_rank_8_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_8_bs64_node65_resnet152.log-Num devices: 16
./16_gpu_rank_8_bs64_node65_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_8_bs64_node65_resnet152.log-Time per mini-batch : 0.6998129653930664
./16_gpu_rank_8_bs64_node65_resnet152.log-Throughput [img/sec] : 1463.2481114791099
./16_gpu_rank_9_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_9_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_9_bs64_node65_resnet152.log---------This process: rank 9--------
./16_gpu_rank_9_bs64_node65_resnet152.log-Num devices: 1
./16_gpu_rank_9_bs64_node65_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_9_bs64_node65_resnet152.log-Time per mini-batch : 0.7016769886016846
./16_gpu_rank_9_bs64_node65_resnet152.log-Throughput [img/sec] : 91.21005967081867
./16_gpu_rank_9_bs64_node65_resnet152.log-
./16_gpu_rank_9_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_9_bs64_node65_resnet152.log-Num devices: 16
./16_gpu_rank_9_bs64_node65_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_9_bs64_node65_resnet152.log-Time per mini-batch : 0.7016769886016846
./16_gpu_rank_9_bs64_node65_resnet152.log-Throughput [img/sec] : 1459.3609547330987
./16_gpu_rank_10_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_10_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_10_bs64_node65_resnet152.log---------This process: rank 10--------
./16_gpu_rank_10_bs64_node65_resnet152.log-Num devices: 1
./16_gpu_rank_10_bs64_node65_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_10_bs64_node65_resnet152.log-Time per mini-batch : 0.7022207713127137
./16_gpu_rank_10_bs64_node65_resnet152.log-Throughput [img/sec] : 91.13942881575552
./16_gpu_rank_10_bs64_node65_resnet152.log-
./16_gpu_rank_10_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_10_bs64_node65_resnet152.log-Num devices: 16
./16_gpu_rank_10_bs64_node65_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_10_bs64_node65_resnet152.log-Time per mini-batch : 0.7022207713127137
./16_gpu_rank_10_bs64_node65_resnet152.log-Throughput [img/sec] : 1458.2308610520884
./16_gpu_rank_11_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./16_gpu_rank_11_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./16_gpu_rank_11_bs64_node65_resnet152.log---------This process: rank 11--------
./16_gpu_rank_11_bs64_node65_resnet152.log-Num devices: 1
./16_gpu_rank_11_bs64_node65_resnet152.log-Mini batch size [img] : 64
./16_gpu_rank_11_bs64_node65_resnet152.log-Time per mini-batch : 0.7035852503776551
./16_gpu_rank_11_bs64_node65_resnet152.log-Throughput [img/sec] : 90.96268002441421
./16_gpu_rank_11_bs64_node65_resnet152.log-
./16_gpu_rank_11_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./16_gpu_rank_11_bs64_node65_resnet152.log-Num devices: 16
./16_gpu_rank_11_bs64_node65_resnet152.log-Mini batch size [img] : 1024
./16_gpu_rank_11_bs64_node65_resnet152.log-Time per mini-batch : 0.7035852503776551
./16_gpu_rank_11_bs64_node65_resnet152.log-Throughput [img/sec] : 1455.4028803906274
./32_gpu_rank_16_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_16_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_16_bs64_node65_resnet152.log---------This process: rank 16--------
./32_gpu_rank_16_bs64_node65_resnet152.log-Num devices: 1
./32_gpu_rank_16_bs64_node65_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_16_bs64_node65_resnet152.log-Time per mini-batch : 0.6962821555137634
./32_gpu_rank_16_bs64_node65_resnet152.log-Throughput [img/sec] : 91.91676031504288
./32_gpu_rank_16_bs64_node65_resnet152.log-
./32_gpu_rank_16_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_16_bs64_node65_resnet152.log-Num devices: 32
./32_gpu_rank_16_bs64_node65_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_16_bs64_node65_resnet152.log-Time per mini-batch : 0.6962821555137634
./32_gpu_rank_16_bs64_node65_resnet152.log-Throughput [img/sec] : 2941.336330081372
./32_gpu_rank_17_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_17_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_17_bs64_node65_resnet152.log---------This process: rank 17--------
./32_gpu_rank_17_bs64_node65_resnet152.log-Num devices: 1
./32_gpu_rank_17_bs64_node65_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_17_bs64_node65_resnet152.log-Time per mini-batch : 0.6921111679077149
./32_gpu_rank_17_bs64_node65_resnet152.log-Throughput [img/sec] : 92.47069396882449
./32_gpu_rank_17_bs64_node65_resnet152.log-
./32_gpu_rank_17_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_17_bs64_node65_resnet152.log-Num devices: 32
./32_gpu_rank_17_bs64_node65_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_17_bs64_node65_resnet152.log-Time per mini-batch : 0.6921111679077149
./32_gpu_rank_17_bs64_node65_resnet152.log-Throughput [img/sec] : 2959.0622070023837
./32_gpu_rank_18_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_18_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_18_bs64_node65_resnet152.log---------This process: rank 18--------
./32_gpu_rank_18_bs64_node65_resnet152.log-Num devices: 1
./32_gpu_rank_18_bs64_node65_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_18_bs64_node65_resnet152.log-Time per mini-batch : 0.694670786857605
./32_gpu_rank_18_bs64_node65_resnet152.log-Throughput [img/sec] : 92.12997179499769
./32_gpu_rank_18_bs64_node65_resnet152.log-
./32_gpu_rank_18_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_18_bs64_node65_resnet152.log-Num devices: 32
./32_gpu_rank_18_bs64_node65_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_18_bs64_node65_resnet152.log-Time per mini-batch : 0.694670786857605
./32_gpu_rank_18_bs64_node65_resnet152.log-Throughput [img/sec] : 2948.159097439926
./32_gpu_rank_19_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_19_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_19_bs64_node65_resnet152.log---------This process: rank 19--------
./32_gpu_rank_19_bs64_node65_resnet152.log-Num devices: 1
./32_gpu_rank_19_bs64_node65_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_19_bs64_node65_resnet152.log-Time per mini-batch : 0.6961898875236511
./32_gpu_rank_19_bs64_node65_resnet152.log-Throughput [img/sec] : 91.92894229999251
./32_gpu_rank_19_bs64_node65_resnet152.log-
./32_gpu_rank_19_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_19_bs64_node65_resnet152.log-Num devices: 32
./32_gpu_rank_19_bs64_node65_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_19_bs64_node65_resnet152.log-Time per mini-batch : 0.6961898875236511
./32_gpu_rank_19_bs64_node65_resnet152.log-Throughput [img/sec] : 2941.7261535997604
./32_gpu_rank_20_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_20_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_20_bs64_node65_resnet152.log---------This process: rank 20--------
./32_gpu_rank_20_bs64_node65_resnet152.log-Num devices: 1
./32_gpu_rank_20_bs64_node65_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_20_bs64_node65_resnet152.log-Time per mini-batch : 0.6927758169174194
./32_gpu_rank_20_bs64_node65_resnet152.log-Throughput [img/sec] : 92.3819775995861
./32_gpu_rank_20_bs64_node65_resnet152.log-
./32_gpu_rank_20_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_20_bs64_node65_resnet152.log-Num devices: 32
./32_gpu_rank_20_bs64_node65_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_20_bs64_node65_resnet152.log-Time per mini-batch : 0.6927758169174194
./32_gpu_rank_20_bs64_node65_resnet152.log-Throughput [img/sec] : 2956.223283186755
./32_gpu_rank_21_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_21_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_21_bs64_node65_resnet152.log---------This process: rank 21--------
./32_gpu_rank_21_bs64_node65_resnet152.log-Num devices: 1
./32_gpu_rank_21_bs64_node65_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_21_bs64_node65_resnet152.log-Time per mini-batch : 0.6958239269256592
./32_gpu_rank_21_bs64_node65_resnet152.log-Throughput [img/sec] : 91.97729127075227
./32_gpu_rank_21_bs64_node65_resnet152.log-
./32_gpu_rank_21_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_21_bs64_node65_resnet152.log-Num devices: 32
./32_gpu_rank_21_bs64_node65_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_21_bs64_node65_resnet152.log-Time per mini-batch : 0.6958239269256592
./32_gpu_rank_21_bs64_node65_resnet152.log-Throughput [img/sec] : 2943.273320664073
./32_gpu_rank_22_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_22_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_22_bs64_node65_resnet152.log---------This process: rank 22--------
./32_gpu_rank_22_bs64_node65_resnet152.log-Num devices: 1
./32_gpu_rank_22_bs64_node65_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_22_bs64_node65_resnet152.log-Time per mini-batch : 0.694828610420227
./32_gpu_rank_22_bs64_node65_resnet152.log-Throughput [img/sec] : 92.10904536773937
./32_gpu_rank_22_bs64_node65_resnet152.log-
./32_gpu_rank_22_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_22_bs64_node65_resnet152.log-Num devices: 32
./32_gpu_rank_22_bs64_node65_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_22_bs64_node65_resnet152.log-Time per mini-batch : 0.694828610420227
./32_gpu_rank_22_bs64_node65_resnet152.log-Throughput [img/sec] : 2947.48945176766
./32_gpu_rank_23_bs64_node65_resnet152.log:--------------------SUMMARY--------------------------
./32_gpu_rank_23_bs64_node65_resnet152.log-Microbenchmark for network : resnet152
./32_gpu_rank_23_bs64_node65_resnet152.log---------This process: rank 23--------
./32_gpu_rank_23_bs64_node65_resnet152.log-Num devices: 1
./32_gpu_rank_23_bs64_node65_resnet152.log-Mini batch size [img] : 64
./32_gpu_rank_23_bs64_node65_resnet152.log-Time per mini-batch : 0.6944083142280578
./32_gpu_rank_23_bs64_node65_resnet152.log-Throughput [img/sec] : 92.16479510494614
./32_gpu_rank_23_bs64_node65_resnet152.log-
./32_gpu_rank_23_bs64_node65_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
./32_gpu_rank_23_bs64_node65_resnet152.log-Num devices: 32
./32_gpu_rank_23_bs64_node65_resnet152.log-Mini batch size [img] : 2048
./32_gpu_rank_23_bs64_node65_resnet152.log-Time per mini-batch : 0.6944083142280578
./32_gpu_rank_23_bs64_node65_resnet152.log-Throughput [img/sec] : 2949.2734433582764
root@BJ-IDC1-10-10-31-65:/data/pytorch_benchmark/log# 

======
    
  log/32_gpu_rank_28_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_28_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/32_gpu_rank_28_bs64_node66_resnet50.log---------This process: rank 28--------
log/32_gpu_rank_28_bs64_node66_resnet50.log-Num devices: 1
log/32_gpu_rank_28_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/32_gpu_rank_28_bs64_node66_resnet50.log-Time per mini-batch : 0.2895250463485718
log/32_gpu_rank_28_bs64_node66_resnet50.log-Throughput [img/sec] : 221.05168726213628
log/32_gpu_rank_28_bs64_node66_resnet50.log-
log/32_gpu_rank_28_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_28_bs64_node66_resnet50.log-Num devices: 32
log/32_gpu_rank_28_bs64_node66_resnet50.log-Mini batch size [img] : 2048
log/32_gpu_rank_28_bs64_node66_resnet50.log-Time per mini-batch : 0.2895250463485718
log/32_gpu_rank_28_bs64_node66_resnet50.log-Throughput [img/sec] : 7073.653992388361
log/32_gpu_rank_27_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_27_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/32_gpu_rank_27_bs64_node66_resnet101.log---------This process: rank 27--------
log/32_gpu_rank_27_bs64_node66_resnet101.log-Num devices: 1
log/32_gpu_rank_27_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/32_gpu_rank_27_bs64_node66_resnet101.log-Time per mini-batch : 0.4927234673500061
log/32_gpu_rank_27_bs64_node66_resnet101.log-Throughput [img/sec] : 129.89030204753288
log/32_gpu_rank_27_bs64_node66_resnet101.log-
log/32_gpu_rank_27_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_27_bs64_node66_resnet101.log-Num devices: 32
log/32_gpu_rank_27_bs64_node66_resnet101.log-Mini batch size [img] : 2048
log/32_gpu_rank_27_bs64_node66_resnet101.log-Time per mini-batch : 0.4927234673500061
log/32_gpu_rank_27_bs64_node66_resnet101.log-Throughput [img/sec] : 4156.489665521052
log/16_gpu_rank_14_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_14_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/16_gpu_rank_14_bs64_node66_resnet101.log---------This process: rank 14--------
log/16_gpu_rank_14_bs64_node66_resnet101.log-Num devices: 1
log/16_gpu_rank_14_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/16_gpu_rank_14_bs64_node66_resnet101.log-Time per mini-batch : 0.4934810924530029
log/16_gpu_rank_14_bs64_node66_resnet101.log-Throughput [img/sec] : 129.69088578828396
log/16_gpu_rank_14_bs64_node66_resnet101.log-
log/16_gpu_rank_14_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_14_bs64_node66_resnet101.log-Num devices: 16
log/16_gpu_rank_14_bs64_node66_resnet101.log-Mini batch size [img] : 1024
log/16_gpu_rank_14_bs64_node66_resnet101.log-Time per mini-batch : 0.4934810924530029
log/16_gpu_rank_14_bs64_node66_resnet101.log-Throughput [img/sec] : 2075.0541726125434
log/8_gpu_rank_7_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/8_gpu_rank_7_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/8_gpu_rank_7_bs64_node66_resnet152.log---------This process: rank 7--------
log/8_gpu_rank_7_bs64_node66_resnet152.log-Num devices: 1
log/8_gpu_rank_7_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/8_gpu_rank_7_bs64_node66_resnet152.log-Time per mini-batch : 0.700956780910492
log/8_gpu_rank_7_bs64_node66_resnet152.log-Throughput [img/sec] : 91.30377470187057
log/8_gpu_rank_7_bs64_node66_resnet152.log-
log/8_gpu_rank_7_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/8_gpu_rank_7_bs64_node66_resnet152.log-Num devices: 8
log/8_gpu_rank_7_bs64_node66_resnet152.log-Mini batch size [img] : 512
log/8_gpu_rank_7_bs64_node66_resnet152.log-Time per mini-batch : 0.700956780910492
log/8_gpu_rank_7_bs64_node66_resnet152.log-Throughput [img/sec] : 730.4301976149645
log/32_gpu_rank_29_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_29_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/32_gpu_rank_29_bs64_node66_resnet152.log---------This process: rank 29--------
log/32_gpu_rank_29_bs64_node66_resnet152.log-Num devices: 1
log/32_gpu_rank_29_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/32_gpu_rank_29_bs64_node66_resnet152.log-Time per mini-batch : 0.6934282279014587
log/32_gpu_rank_29_bs64_node66_resnet152.log-Throughput [img/sec] : 92.29506014441465
log/32_gpu_rank_29_bs64_node66_resnet152.log-
log/32_gpu_rank_29_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_29_bs64_node66_resnet152.log-Num devices: 32
log/32_gpu_rank_29_bs64_node66_resnet152.log-Mini batch size [img] : 2048
log/32_gpu_rank_29_bs64_node66_resnet152.log-Time per mini-batch : 0.6934282279014587
log/32_gpu_rank_29_bs64_node66_resnet152.log-Throughput [img/sec] : 2953.4419246212688
log/32_gpu_rank_31_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_31_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/32_gpu_rank_31_bs64_node66_resnet152.log---------This process: rank 31--------
log/32_gpu_rank_31_bs64_node66_resnet152.log-Num devices: 1
log/32_gpu_rank_31_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/32_gpu_rank_31_bs64_node66_resnet152.log-Time per mini-batch : 0.694501187801361
log/32_gpu_rank_31_bs64_node66_resnet152.log-Throughput [img/sec] : 92.15247018167098
log/32_gpu_rank_31_bs64_node66_resnet152.log-
log/32_gpu_rank_31_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_31_bs64_node66_resnet152.log-Num devices: 32
log/32_gpu_rank_31_bs64_node66_resnet152.log-Mini batch size [img] : 2048
log/32_gpu_rank_31_bs64_node66_resnet152.log-Time per mini-batch : 0.694501187801361
log/32_gpu_rank_31_bs64_node66_resnet152.log-Throughput [img/sec] : 2948.8790458134713
log/16_gpu_rank_13_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_13_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/16_gpu_rank_13_bs64_node66_resnet50.log---------This process: rank 13--------
log/16_gpu_rank_13_bs64_node66_resnet50.log-Num devices: 1
log/16_gpu_rank_13_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/16_gpu_rank_13_bs64_node66_resnet50.log-Time per mini-batch : 0.2849256157875061
log/16_gpu_rank_13_bs64_node66_resnet50.log-Throughput [img/sec] : 224.6200287156013
log/16_gpu_rank_13_bs64_node66_resnet50.log-
log/16_gpu_rank_13_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_13_bs64_node66_resnet50.log-Num devices: 16
log/16_gpu_rank_13_bs64_node66_resnet50.log-Mini batch size [img] : 1024
log/16_gpu_rank_13_bs64_node66_resnet50.log-Time per mini-batch : 0.2849256157875061
log/16_gpu_rank_13_bs64_node66_resnet50.log-Throughput [img/sec] : 3593.920459449621
log/16_gpu_rank_12_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_12_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/16_gpu_rank_12_bs64_node66_resnet152.log---------This process: rank 12--------
log/16_gpu_rank_12_bs64_node66_resnet152.log-Num devices: 1
log/16_gpu_rank_12_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/16_gpu_rank_12_bs64_node66_resnet152.log-Time per mini-batch : 0.7029510426521302
log/16_gpu_rank_12_bs64_node66_resnet152.log-Throughput [img/sec] : 91.04474723949122
log/16_gpu_rank_12_bs64_node66_resnet152.log-
log/16_gpu_rank_12_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_12_bs64_node66_resnet152.log-Num devices: 16
log/16_gpu_rank_12_bs64_node66_resnet152.log-Mini batch size [img] : 1024
log/16_gpu_rank_12_bs64_node66_resnet152.log-Time per mini-batch : 0.7029510426521302
log/16_gpu_rank_12_bs64_node66_resnet152.log-Throughput [img/sec] : 1456.7159558318594
log/16_gpu_rank_15_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_15_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/16_gpu_rank_15_bs64_node66_resnet101.log---------This process: rank 15--------
log/16_gpu_rank_15_bs64_node66_resnet101.log-Num devices: 1
log/16_gpu_rank_15_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/16_gpu_rank_15_bs64_node66_resnet101.log-Time per mini-batch : 0.4963023328781128
log/16_gpu_rank_15_bs64_node66_resnet101.log-Throughput [img/sec] : 128.9536553835176
log/16_gpu_rank_15_bs64_node66_resnet101.log-
log/16_gpu_rank_15_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_15_bs64_node66_resnet101.log-Num devices: 16
log/16_gpu_rank_15_bs64_node66_resnet101.log-Mini batch size [img] : 1024
log/16_gpu_rank_15_bs64_node66_resnet101.log-Time per mini-batch : 0.4963023328781128
log/16_gpu_rank_15_bs64_node66_resnet101.log-Throughput [img/sec] : 2063.2584861362816
log/16_gpu_rank_13_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_13_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/16_gpu_rank_13_bs64_node66_resnet101.log---------This process: rank 13--------
log/16_gpu_rank_13_bs64_node66_resnet101.log-Num devices: 1
log/16_gpu_rank_13_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/16_gpu_rank_13_bs64_node66_resnet101.log-Time per mini-batch : 0.49439082860946654
log/16_gpu_rank_13_bs64_node66_resnet101.log-Throughput [img/sec] : 129.4522395975825
log/16_gpu_rank_13_bs64_node66_resnet101.log-
log/16_gpu_rank_13_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_13_bs64_node66_resnet101.log-Num devices: 16
log/16_gpu_rank_13_bs64_node66_resnet101.log-Mini batch size [img] : 1024
log/16_gpu_rank_13_bs64_node66_resnet101.log-Time per mini-batch : 0.49439082860946654
log/16_gpu_rank_13_bs64_node66_resnet101.log-Throughput [img/sec] : 2071.23583356132
log/32_gpu_rank_30_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_30_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/32_gpu_rank_30_bs64_node66_resnet50.log---------This process: rank 30--------
log/32_gpu_rank_30_bs64_node66_resnet50.log-Num devices: 1
log/32_gpu_rank_30_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/32_gpu_rank_30_bs64_node66_resnet50.log-Time per mini-batch : 0.29039337635040285
log/32_gpu_rank_30_bs64_node66_resnet50.log-Throughput [img/sec] : 220.39070175889435
log/32_gpu_rank_30_bs64_node66_resnet50.log-
log/32_gpu_rank_30_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_30_bs64_node66_resnet50.log-Num devices: 32
log/32_gpu_rank_30_bs64_node66_resnet50.log-Mini batch size [img] : 2048
log/32_gpu_rank_30_bs64_node66_resnet50.log-Time per mini-batch : 0.29039337635040285
log/32_gpu_rank_30_bs64_node66_resnet50.log-Throughput [img/sec] : 7052.502456284619
log/32_gpu_rank_24_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_24_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/32_gpu_rank_24_bs64_node66_resnet152.log---------This process: rank 24--------
log/32_gpu_rank_24_bs64_node66_resnet152.log-Num devices: 1
log/32_gpu_rank_24_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/32_gpu_rank_24_bs64_node66_resnet152.log-Time per mini-batch : 0.6944075989723205
log/32_gpu_rank_24_bs64_node66_resnet152.log-Throughput [img/sec] : 92.16489003679679
log/32_gpu_rank_24_bs64_node66_resnet152.log-
log/32_gpu_rank_24_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_24_bs64_node66_resnet152.log-Num devices: 32
log/32_gpu_rank_24_bs64_node66_resnet152.log-Mini batch size [img] : 2048
log/32_gpu_rank_24_bs64_node66_resnet152.log-Time per mini-batch : 0.6944075989723205
log/32_gpu_rank_24_bs64_node66_resnet152.log-Throughput [img/sec] : 2949.2764811774973
log/4_gpu_rank_3_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/4_gpu_rank_3_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/4_gpu_rank_3_bs64_node66_resnet152.log---------This process: rank 3--------
log/4_gpu_rank_3_bs64_node66_resnet152.log-Num devices: 1
log/4_gpu_rank_3_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/4_gpu_rank_3_bs64_node66_resnet152.log-Time per mini-batch : 0.7021686434745789
log/4_gpu_rank_3_bs64_node66_resnet152.log-Throughput [img/sec] : 91.14619485613223
log/4_gpu_rank_3_bs64_node66_resnet152.log-
log/4_gpu_rank_3_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/4_gpu_rank_3_bs64_node66_resnet152.log-Num devices: 4
log/4_gpu_rank_3_bs64_node66_resnet152.log-Mini batch size [img] : 256
log/4_gpu_rank_3_bs64_node66_resnet152.log-Time per mini-batch : 0.7021686434745789
log/4_gpu_rank_3_bs64_node66_resnet152.log-Throughput [img/sec] : 364.5847794245289
log/16_gpu_rank_12_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_12_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/16_gpu_rank_12_bs64_node66_resnet50.log---------This process: rank 12--------
log/16_gpu_rank_12_bs64_node66_resnet50.log-Num devices: 1
log/16_gpu_rank_12_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/16_gpu_rank_12_bs64_node66_resnet50.log-Time per mini-batch : 0.2852110815048218
log/16_gpu_rank_12_bs64_node66_resnet50.log-Throughput [img/sec] : 224.39520814663024
log/16_gpu_rank_12_bs64_node66_resnet50.log-
log/16_gpu_rank_12_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_12_bs64_node66_resnet50.log-Num devices: 16
log/16_gpu_rank_12_bs64_node66_resnet50.log-Mini batch size [img] : 1024
log/16_gpu_rank_12_bs64_node66_resnet50.log-Time per mini-batch : 0.2852110815048218
log/16_gpu_rank_12_bs64_node66_resnet50.log-Throughput [img/sec] : 3590.323330346084
log/8_gpu_rank_6_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/8_gpu_rank_6_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/8_gpu_rank_6_bs64_node66_resnet50.log---------This process: rank 6--------
log/8_gpu_rank_6_bs64_node66_resnet50.log-Num devices: 1
log/8_gpu_rank_6_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/8_gpu_rank_6_bs64_node66_resnet50.log-Time per mini-batch : 0.28695858716964723
log/8_gpu_rank_6_bs64_node66_resnet50.log-Throughput [img/sec] : 223.02869773387823
log/8_gpu_rank_6_bs64_node66_resnet50.log-
log/8_gpu_rank_6_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/8_gpu_rank_6_bs64_node66_resnet50.log-Num devices: 8
log/8_gpu_rank_6_bs64_node66_resnet50.log-Mini batch size [img] : 512
log/8_gpu_rank_6_bs64_node66_resnet50.log-Time per mini-batch : 0.28695858716964723
log/8_gpu_rank_6_bs64_node66_resnet50.log-Throughput [img/sec] : 1784.2295818710259
log/16_gpu_rank_14_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_14_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/16_gpu_rank_14_bs64_node66_resnet152.log---------This process: rank 14--------
log/16_gpu_rank_14_bs64_node66_resnet152.log-Num devices: 1
log/16_gpu_rank_14_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/16_gpu_rank_14_bs64_node66_resnet152.log-Time per mini-batch : 0.7007429075241088
log/16_gpu_rank_14_bs64_node66_resnet152.log-Throughput [img/sec] : 91.33164148050703
log/16_gpu_rank_14_bs64_node66_resnet152.log-
log/16_gpu_rank_14_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_14_bs64_node66_resnet152.log-Num devices: 16
log/16_gpu_rank_14_bs64_node66_resnet152.log-Mini batch size [img] : 1024
log/16_gpu_rank_14_bs64_node66_resnet152.log-Time per mini-batch : 0.7007429075241088
log/16_gpu_rank_14_bs64_node66_resnet152.log-Throughput [img/sec] : 1461.3062636881125
log/16_gpu_rank_15_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_15_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/16_gpu_rank_15_bs64_node66_resnet152.log---------This process: rank 15--------
log/16_gpu_rank_15_bs64_node66_resnet152.log-Num devices: 1
log/16_gpu_rank_15_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/16_gpu_rank_15_bs64_node66_resnet152.log-Time per mini-batch : 0.7015455627441406
log/16_gpu_rank_15_bs64_node66_resnet152.log-Throughput [img/sec] : 91.22714674391194
log/16_gpu_rank_15_bs64_node66_resnet152.log-
log/16_gpu_rank_15_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_15_bs64_node66_resnet152.log-Num devices: 16
log/16_gpu_rank_15_bs64_node66_resnet152.log-Mini batch size [img] : 1024
log/16_gpu_rank_15_bs64_node66_resnet152.log-Time per mini-batch : 0.7015455627441406
log/16_gpu_rank_15_bs64_node66_resnet152.log-Throughput [img/sec] : 1459.6343479025911
log/8_gpu_rank_7_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/8_gpu_rank_7_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/8_gpu_rank_7_bs64_node66_resnet50.log---------This process: rank 7--------
log/8_gpu_rank_7_bs64_node66_resnet50.log-Num devices: 1
log/8_gpu_rank_7_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/8_gpu_rank_7_bs64_node66_resnet50.log-Time per mini-batch : 0.2871426248550415
log/8_gpu_rank_7_bs64_node66_resnet50.log-Throughput [img/sec] : 222.88575244552837
log/8_gpu_rank_7_bs64_node66_resnet50.log-
log/8_gpu_rank_7_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/8_gpu_rank_7_bs64_node66_resnet50.log-Num devices: 8
log/8_gpu_rank_7_bs64_node66_resnet50.log-Mini batch size [img] : 512
log/8_gpu_rank_7_bs64_node66_resnet50.log-Time per mini-batch : 0.2871426248550415
log/8_gpu_rank_7_bs64_node66_resnet50.log-Throughput [img/sec] : 1783.086019564227
log/4_gpu_rank_3_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/4_gpu_rank_3_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/4_gpu_rank_3_bs64_node66_resnet50.log---------This process: rank 3--------
log/4_gpu_rank_3_bs64_node66_resnet50.log-Num devices: 1
log/4_gpu_rank_3_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/4_gpu_rank_3_bs64_node66_resnet50.log-Time per mini-batch : 0.2913595008850098
log/4_gpu_rank_3_bs64_node66_resnet50.log-Throughput [img/sec] : 219.65990402097353
log/4_gpu_rank_3_bs64_node66_resnet50.log-
log/4_gpu_rank_3_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/4_gpu_rank_3_bs64_node66_resnet50.log-Num devices: 4
log/4_gpu_rank_3_bs64_node66_resnet50.log-Mini batch size [img] : 256
log/4_gpu_rank_3_bs64_node66_resnet50.log-Time per mini-batch : 0.2913595008850098
log/4_gpu_rank_3_bs64_node66_resnet50.log-Throughput [img/sec] : 878.6396160838941
log/32_gpu_rank_25_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_25_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/32_gpu_rank_25_bs64_node66_resnet152.log---------This process: rank 25--------
log/32_gpu_rank_25_bs64_node66_resnet152.log-Num devices: 1
log/32_gpu_rank_25_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/32_gpu_rank_25_bs64_node66_resnet152.log-Time per mini-batch : 0.693851170539856
log/32_gpu_rank_25_bs64_node66_resnet152.log-Throughput [img/sec] : 92.23880093796531
log/32_gpu_rank_25_bs64_node66_resnet152.log-
log/32_gpu_rank_25_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_25_bs64_node66_resnet152.log-Num devices: 32
log/32_gpu_rank_25_bs64_node66_resnet152.log-Mini batch size [img] : 2048
log/32_gpu_rank_25_bs64_node66_resnet152.log-Time per mini-batch : 0.693851170539856
log/32_gpu_rank_25_bs64_node66_resnet152.log-Throughput [img/sec] : 2951.64163001489
log/32_gpu_rank_28_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_28_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/32_gpu_rank_28_bs64_node66_resnet152.log---------This process: rank 28--------
log/32_gpu_rank_28_bs64_node66_resnet152.log-Num devices: 1
log/32_gpu_rank_28_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/32_gpu_rank_28_bs64_node66_resnet152.log-Time per mini-batch : 0.6942754197120666
log/32_gpu_rank_28_bs64_node66_resnet152.log-Throughput [img/sec] : 92.18243680086269
log/32_gpu_rank_28_bs64_node66_resnet152.log-
log/32_gpu_rank_28_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_28_bs64_node66_resnet152.log-Num devices: 32
log/32_gpu_rank_28_bs64_node66_resnet152.log-Mini batch size [img] : 2048
log/32_gpu_rank_28_bs64_node66_resnet152.log-Time per mini-batch : 0.6942754197120666
log/32_gpu_rank_28_bs64_node66_resnet152.log-Throughput [img/sec] : 2949.837977627606
log/32_gpu_rank_27_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_27_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/32_gpu_rank_27_bs64_node66_resnet152.log---------This process: rank 27--------
log/32_gpu_rank_27_bs64_node66_resnet152.log-Num devices: 1
log/32_gpu_rank_27_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/32_gpu_rank_27_bs64_node66_resnet152.log-Time per mini-batch : 0.6961368918418884
log/32_gpu_rank_27_bs64_node66_resnet152.log-Throughput [img/sec] : 91.93594068928635
log/32_gpu_rank_27_bs64_node66_resnet152.log-
log/32_gpu_rank_27_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_27_bs64_node66_resnet152.log-Num devices: 32
log/32_gpu_rank_27_bs64_node66_resnet152.log-Mini batch size [img] : 2048
log/32_gpu_rank_27_bs64_node66_resnet152.log-Time per mini-batch : 0.6961368918418884
log/32_gpu_rank_27_bs64_node66_resnet152.log-Throughput [img/sec] : 2941.9501020571633
log/16_gpu_rank_15_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_15_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/16_gpu_rank_15_bs64_node66_resnet50.log---------This process: rank 15--------
log/16_gpu_rank_15_bs64_node66_resnet50.log-Num devices: 1
log/16_gpu_rank_15_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/16_gpu_rank_15_bs64_node66_resnet50.log-Time per mini-batch : 0.2873330402374268
log/16_gpu_rank_15_bs64_node66_resnet50.log-Throughput [img/sec] : 222.7380462306598
log/16_gpu_rank_15_bs64_node66_resnet50.log-
log/16_gpu_rank_15_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_15_bs64_node66_resnet50.log-Num devices: 16
log/16_gpu_rank_15_bs64_node66_resnet50.log-Mini batch size [img] : 1024
log/16_gpu_rank_15_bs64_node66_resnet50.log-Time per mini-batch : 0.2873330402374268
log/16_gpu_rank_15_bs64_node66_resnet50.log-Throughput [img/sec] : 3563.8087396905566
log/32_gpu_rank_31_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_31_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/32_gpu_rank_31_bs64_node66_resnet50.log---------This process: rank 31--------
log/32_gpu_rank_31_bs64_node66_resnet50.log-Num devices: 1
log/32_gpu_rank_31_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/32_gpu_rank_31_bs64_node66_resnet50.log-Time per mini-batch : 0.2919183611869812
log/32_gpu_rank_31_bs64_node66_resnet50.log-Throughput [img/sec] : 219.2393782281011
log/32_gpu_rank_31_bs64_node66_resnet50.log-
log/32_gpu_rank_31_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_31_bs64_node66_resnet50.log-Num devices: 32
log/32_gpu_rank_31_bs64_node66_resnet50.log-Mini batch size [img] : 2048
log/32_gpu_rank_31_bs64_node66_resnet50.log-Time per mini-batch : 0.2919183611869812
log/32_gpu_rank_31_bs64_node66_resnet50.log-Throughput [img/sec] : 7015.660103299235
log/16_gpu_rank_12_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_12_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/16_gpu_rank_12_bs64_node66_resnet101.log---------This process: rank 12--------
log/16_gpu_rank_12_bs64_node66_resnet101.log-Num devices: 1
log/16_gpu_rank_12_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/16_gpu_rank_12_bs64_node66_resnet101.log-Time per mini-batch : 0.49412367343902586
log/16_gpu_rank_12_bs64_node66_resnet101.log-Throughput [img/sec] : 129.5222298388776
log/16_gpu_rank_12_bs64_node66_resnet101.log-
log/16_gpu_rank_12_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_12_bs64_node66_resnet101.log-Num devices: 16
log/16_gpu_rank_12_bs64_node66_resnet101.log-Mini batch size [img] : 1024
log/16_gpu_rank_12_bs64_node66_resnet101.log-Time per mini-batch : 0.49412367343902586
log/16_gpu_rank_12_bs64_node66_resnet101.log-Throughput [img/sec] : 2072.3556774220415
log/32_gpu_rank_24_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_24_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/32_gpu_rank_24_bs64_node66_resnet101.log---------This process: rank 24--------
log/32_gpu_rank_24_bs64_node66_resnet101.log-Num devices: 1
log/32_gpu_rank_24_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/32_gpu_rank_24_bs64_node66_resnet101.log-Time per mini-batch : 0.488249614238739
log/32_gpu_rank_24_bs64_node66_resnet101.log-Throughput [img/sec] : 131.08049271024302
log/32_gpu_rank_24_bs64_node66_resnet101.log-
log/32_gpu_rank_24_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_24_bs64_node66_resnet101.log-Num devices: 32
log/32_gpu_rank_24_bs64_node66_resnet101.log-Mini batch size [img] : 2048
log/32_gpu_rank_24_bs64_node66_resnet101.log-Time per mini-batch : 0.488249614238739
log/32_gpu_rank_24_bs64_node66_resnet101.log-Throughput [img/sec] : 4194.575766727777
log/32_gpu_rank_29_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_29_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/32_gpu_rank_29_bs64_node66_resnet101.log---------This process: rank 29--------
log/32_gpu_rank_29_bs64_node66_resnet101.log-Num devices: 1
log/32_gpu_rank_29_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/32_gpu_rank_29_bs64_node66_resnet101.log-Time per mini-batch : 0.4935885524749756
log/32_gpu_rank_29_bs64_node66_resnet101.log-Throughput [img/sec] : 129.66265056004258
log/32_gpu_rank_29_bs64_node66_resnet101.log-
log/32_gpu_rank_29_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_29_bs64_node66_resnet101.log-Num devices: 32
log/32_gpu_rank_29_bs64_node66_resnet101.log-Mini batch size [img] : 2048
log/32_gpu_rank_29_bs64_node66_resnet101.log-Time per mini-batch : 0.4935885524749756
log/32_gpu_rank_29_bs64_node66_resnet101.log-Throughput [img/sec] : 4149.2048179213625
log/8_gpu_rank_7_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/8_gpu_rank_7_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/8_gpu_rank_7_bs64_node66_resnet101.log---------This process: rank 7--------
log/8_gpu_rank_7_bs64_node66_resnet101.log-Num devices: 1
log/8_gpu_rank_7_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/8_gpu_rank_7_bs64_node66_resnet101.log-Time per mini-batch : 0.4804459285736084
log/8_gpu_rank_7_bs64_node66_resnet101.log-Throughput [img/sec] : 133.20957925485814
log/8_gpu_rank_7_bs64_node66_resnet101.log-
log/8_gpu_rank_7_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/8_gpu_rank_7_bs64_node66_resnet101.log-Num devices: 8
log/8_gpu_rank_7_bs64_node66_resnet101.log-Mini batch size [img] : 512
log/8_gpu_rank_7_bs64_node66_resnet101.log-Time per mini-batch : 0.4804459285736084
log/8_gpu_rank_7_bs64_node66_resnet101.log-Throughput [img/sec] : 1065.6766340388651
log/32_gpu_rank_24_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_24_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/32_gpu_rank_24_bs64_node66_resnet50.log---------This process: rank 24--------
log/32_gpu_rank_24_bs64_node66_resnet50.log-Num devices: 1
log/32_gpu_rank_24_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/32_gpu_rank_24_bs64_node66_resnet50.log-Time per mini-batch : 0.2919481873512268
log/32_gpu_rank_24_bs64_node66_resnet50.log-Throughput [img/sec] : 219.21698017944917
log/32_gpu_rank_24_bs64_node66_resnet50.log-
log/32_gpu_rank_24_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_24_bs64_node66_resnet50.log-Num devices: 32
log/32_gpu_rank_24_bs64_node66_resnet50.log-Mini batch size [img] : 2048
log/32_gpu_rank_24_bs64_node66_resnet50.log-Time per mini-batch : 0.2919481873512268
log/32_gpu_rank_24_bs64_node66_resnet50.log-Throughput [img/sec] : 7014.943365742373
log/32_gpu_rank_30_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_30_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/32_gpu_rank_30_bs64_node66_resnet101.log---------This process: rank 30--------
log/32_gpu_rank_30_bs64_node66_resnet101.log-Num devices: 1
log/32_gpu_rank_30_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/32_gpu_rank_30_bs64_node66_resnet101.log-Time per mini-batch : 0.4919541883468628
log/32_gpu_rank_30_bs64_node66_resnet101.log-Throughput [img/sec] : 130.09341421619413
log/32_gpu_rank_30_bs64_node66_resnet101.log-
log/32_gpu_rank_30_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_30_bs64_node66_resnet101.log-Num devices: 32
log/32_gpu_rank_30_bs64_node66_resnet101.log-Mini batch size [img] : 2048
log/32_gpu_rank_30_bs64_node66_resnet101.log-Time per mini-batch : 0.4919541883468628
log/32_gpu_rank_30_bs64_node66_resnet101.log-Throughput [img/sec] : 4162.989254918212
log/4_gpu_rank_3_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/4_gpu_rank_3_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/4_gpu_rank_3_bs64_node66_resnet101.log---------This process: rank 3--------
log/4_gpu_rank_3_bs64_node66_resnet101.log-Num devices: 1
log/4_gpu_rank_3_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/4_gpu_rank_3_bs64_node66_resnet101.log-Time per mini-batch : 0.4907761836051941
log/4_gpu_rank_3_bs64_node66_resnet101.log-Throughput [img/sec] : 130.4056760249901
log/4_gpu_rank_3_bs64_node66_resnet101.log-
log/4_gpu_rank_3_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/4_gpu_rank_3_bs64_node66_resnet101.log-Num devices: 4
log/4_gpu_rank_3_bs64_node66_resnet101.log-Mini batch size [img] : 256
log/4_gpu_rank_3_bs64_node66_resnet101.log-Time per mini-batch : 0.4907761836051941
log/4_gpu_rank_3_bs64_node66_resnet101.log-Throughput [img/sec] : 521.6227040999604
log/8_gpu_rank_6_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/8_gpu_rank_6_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/8_gpu_rank_6_bs64_node66_resnet152.log---------This process: rank 6--------
log/8_gpu_rank_6_bs64_node66_resnet152.log-Num devices: 1
log/8_gpu_rank_6_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/8_gpu_rank_6_bs64_node66_resnet152.log-Time per mini-batch : 0.7002823305130005
log/8_gpu_rank_6_bs64_node66_resnet152.log-Throughput [img/sec] : 91.39171047356858
log/8_gpu_rank_6_bs64_node66_resnet152.log-
log/8_gpu_rank_6_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/8_gpu_rank_6_bs64_node66_resnet152.log-Num devices: 8
log/8_gpu_rank_6_bs64_node66_resnet152.log-Mini batch size [img] : 512
log/8_gpu_rank_6_bs64_node66_resnet152.log-Time per mini-batch : 0.7002823305130005
log/8_gpu_rank_6_bs64_node66_resnet152.log-Throughput [img/sec] : 731.1336837885486
log/32_gpu_rank_26_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_26_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/32_gpu_rank_26_bs64_node66_resnet50.log---------This process: rank 26--------
log/32_gpu_rank_26_bs64_node66_resnet50.log-Num devices: 1
log/32_gpu_rank_26_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/32_gpu_rank_26_bs64_node66_resnet50.log-Time per mini-batch : 0.2902357459068298
log/32_gpu_rank_26_bs64_node66_resnet50.log-Throughput [img/sec] : 220.51039853838333
log/32_gpu_rank_26_bs64_node66_resnet50.log-
log/32_gpu_rank_26_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_26_bs64_node66_resnet50.log-Num devices: 32
log/32_gpu_rank_26_bs64_node66_resnet50.log-Mini batch size [img] : 2048
log/32_gpu_rank_26_bs64_node66_resnet50.log-Time per mini-batch : 0.2902357459068298
log/32_gpu_rank_26_bs64_node66_resnet50.log-Throughput [img/sec] : 7056.3327532282665
log/32_gpu_rank_26_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_26_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/32_gpu_rank_26_bs64_node66_resnet101.log---------This process: rank 26--------
log/32_gpu_rank_26_bs64_node66_resnet101.log-Num devices: 1
log/32_gpu_rank_26_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/32_gpu_rank_26_bs64_node66_resnet101.log-Time per mini-batch : 0.49259581804275515
log/32_gpu_rank_26_bs64_node66_resnet101.log-Throughput [img/sec] : 129.9239613001446
log/32_gpu_rank_26_bs64_node66_resnet101.log-
log/32_gpu_rank_26_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_26_bs64_node66_resnet101.log-Num devices: 32
log/32_gpu_rank_26_bs64_node66_resnet101.log-Mini batch size [img] : 2048
log/32_gpu_rank_26_bs64_node66_resnet101.log-Time per mini-batch : 0.49259581804275515
log/32_gpu_rank_26_bs64_node66_resnet101.log-Throughput [img/sec] : 4157.566761604628
log/16_gpu_rank_13_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_13_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/16_gpu_rank_13_bs64_node66_resnet152.log---------This process: rank 13--------
log/16_gpu_rank_13_bs64_node66_resnet152.log-Num devices: 1
log/16_gpu_rank_13_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/16_gpu_rank_13_bs64_node66_resnet152.log-Time per mini-batch : 0.7023593521118164
log/16_gpu_rank_13_bs64_node66_resnet152.log-Throughput [img/sec] : 91.12144631885121
log/16_gpu_rank_13_bs64_node66_resnet152.log-
log/16_gpu_rank_13_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_13_bs64_node66_resnet152.log-Num devices: 16
log/16_gpu_rank_13_bs64_node66_resnet152.log-Mini batch size [img] : 1024
log/16_gpu_rank_13_bs64_node66_resnet152.log-Time per mini-batch : 0.7023593521118164
log/16_gpu_rank_13_bs64_node66_resnet152.log-Throughput [img/sec] : 1457.9431411016194
log/32_gpu_rank_27_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_27_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/32_gpu_rank_27_bs64_node66_resnet50.log---------This process: rank 27--------
log/32_gpu_rank_27_bs64_node66_resnet50.log-Num devices: 1
log/32_gpu_rank_27_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/32_gpu_rank_27_bs64_node66_resnet50.log-Time per mini-batch : 0.2932941222190857
log/32_gpu_rank_27_bs64_node66_resnet50.log-Throughput [img/sec] : 218.21098737257714
log/32_gpu_rank_27_bs64_node66_resnet50.log-
log/32_gpu_rank_27_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_27_bs64_node66_resnet50.log-Num devices: 32
log/32_gpu_rank_27_bs64_node66_resnet50.log-Mini batch size [img] : 2048
log/32_gpu_rank_27_bs64_node66_resnet50.log-Time per mini-batch : 0.2932941222190857
log/32_gpu_rank_27_bs64_node66_resnet50.log-Throughput [img/sec] : 6982.751595922468
log/32_gpu_rank_31_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_31_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/32_gpu_rank_31_bs64_node66_resnet101.log---------This process: rank 31--------
log/32_gpu_rank_31_bs64_node66_resnet101.log-Num devices: 1
log/32_gpu_rank_31_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/32_gpu_rank_31_bs64_node66_resnet101.log-Time per mini-batch : 0.4902569341659546
log/32_gpu_rank_31_bs64_node66_resnet101.log-Throughput [img/sec] : 130.54379354955876
log/32_gpu_rank_31_bs64_node66_resnet101.log-
log/32_gpu_rank_31_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_31_bs64_node66_resnet101.log-Num devices: 32
log/32_gpu_rank_31_bs64_node66_resnet101.log-Mini batch size [img] : 2048
log/32_gpu_rank_31_bs64_node66_resnet101.log-Time per mini-batch : 0.4902569341659546
log/32_gpu_rank_31_bs64_node66_resnet101.log-Throughput [img/sec] : 4177.40139358588
log/32_gpu_rank_28_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_28_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/32_gpu_rank_28_bs64_node66_resnet101.log---------This process: rank 28--------
log/32_gpu_rank_28_bs64_node66_resnet101.log-Num devices: 1
log/32_gpu_rank_28_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/32_gpu_rank_28_bs64_node66_resnet101.log-Time per mini-batch : 0.492471010684967
log/32_gpu_rank_28_bs64_node66_resnet101.log-Throughput [img/sec] : 129.9568880429811
log/32_gpu_rank_28_bs64_node66_resnet101.log-
log/32_gpu_rank_28_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_28_bs64_node66_resnet101.log-Num devices: 32
log/32_gpu_rank_28_bs64_node66_resnet101.log-Mini batch size [img] : 2048
log/32_gpu_rank_28_bs64_node66_resnet101.log-Time per mini-batch : 0.492471010684967
log/32_gpu_rank_28_bs64_node66_resnet101.log-Throughput [img/sec] : 4158.620417375395
log/8_gpu_rank_6_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/8_gpu_rank_6_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/8_gpu_rank_6_bs64_node66_resnet101.log---------This process: rank 6--------
log/8_gpu_rank_6_bs64_node66_resnet101.log-Num devices: 1
log/8_gpu_rank_6_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/8_gpu_rank_6_bs64_node66_resnet101.log-Time per mini-batch : 0.4808246350288391
log/8_gpu_rank_6_bs64_node66_resnet101.log-Throughput [img/sec] : 133.10466090440931
log/8_gpu_rank_6_bs64_node66_resnet101.log-
log/8_gpu_rank_6_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/8_gpu_rank_6_bs64_node66_resnet101.log-Num devices: 8
log/8_gpu_rank_6_bs64_node66_resnet101.log-Mini batch size [img] : 512
log/8_gpu_rank_6_bs64_node66_resnet101.log-Time per mini-batch : 0.4808246350288391
log/8_gpu_rank_6_bs64_node66_resnet101.log-Throughput [img/sec] : 1064.8372872352745
log/16_gpu_rank_14_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/16_gpu_rank_14_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/16_gpu_rank_14_bs64_node66_resnet50.log---------This process: rank 14--------
log/16_gpu_rank_14_bs64_node66_resnet50.log-Num devices: 1
log/16_gpu_rank_14_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/16_gpu_rank_14_bs64_node66_resnet50.log-Time per mini-batch : 0.2851530408859253
log/16_gpu_rank_14_bs64_node66_resnet50.log-Throughput [img/sec] : 224.44088199502326
log/16_gpu_rank_14_bs64_node66_resnet50.log-
log/16_gpu_rank_14_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/16_gpu_rank_14_bs64_node66_resnet50.log-Num devices: 16
log/16_gpu_rank_14_bs64_node66_resnet50.log-Mini batch size [img] : 1024
log/16_gpu_rank_14_bs64_node66_resnet50.log-Time per mini-batch : 0.2851530408859253
log/16_gpu_rank_14_bs64_node66_resnet50.log-Throughput [img/sec] : 3591.054111920372
log/32_gpu_rank_26_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_26_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/32_gpu_rank_26_bs64_node66_resnet152.log---------This process: rank 26--------
log/32_gpu_rank_26_bs64_node66_resnet152.log-Num devices: 1
log/32_gpu_rank_26_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/32_gpu_rank_26_bs64_node66_resnet152.log-Time per mini-batch : 0.6952880001068116
log/32_gpu_rank_26_bs64_node66_resnet152.log-Throughput [img/sec] : 92.04818721187219
log/32_gpu_rank_26_bs64_node66_resnet152.log-
log/32_gpu_rank_26_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_26_bs64_node66_resnet152.log-Num devices: 32
log/32_gpu_rank_26_bs64_node66_resnet152.log-Mini batch size [img] : 2048
log/32_gpu_rank_26_bs64_node66_resnet152.log-Time per mini-batch : 0.6952880001068116
log/32_gpu_rank_26_bs64_node66_resnet152.log-Throughput [img/sec] : 2945.54199077991
log/32_gpu_rank_25_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_25_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/32_gpu_rank_25_bs64_node66_resnet50.log---------This process: rank 25--------
log/32_gpu_rank_25_bs64_node66_resnet50.log-Num devices: 1
log/32_gpu_rank_25_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/32_gpu_rank_25_bs64_node66_resnet50.log-Time per mini-batch : 0.29191376447677614
log/32_gpu_rank_25_bs64_node66_resnet50.log-Throughput [img/sec] : 219.24283054864878
log/32_gpu_rank_25_bs64_node66_resnet50.log-
log/32_gpu_rank_25_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_25_bs64_node66_resnet50.log-Num devices: 32
log/32_gpu_rank_25_bs64_node66_resnet50.log-Mini batch size [img] : 2048
log/32_gpu_rank_25_bs64_node66_resnet50.log-Time per mini-batch : 0.29191376447677614
log/32_gpu_rank_25_bs64_node66_resnet50.log-Throughput [img/sec] : 7015.770577556761
log/32_gpu_rank_30_bs64_node66_resnet152.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_30_bs64_node66_resnet152.log-Microbenchmark for network : resnet152
log/32_gpu_rank_30_bs64_node66_resnet152.log---------This process: rank 30--------
log/32_gpu_rank_30_bs64_node66_resnet152.log-Num devices: 1
log/32_gpu_rank_30_bs64_node66_resnet152.log-Mini batch size [img] : 64
log/32_gpu_rank_30_bs64_node66_resnet152.log-Time per mini-batch : 0.6958770966529846
log/32_gpu_rank_30_bs64_node66_resnet152.log-Throughput [img/sec] : 91.97026358221285
log/32_gpu_rank_30_bs64_node66_resnet152.log-
log/32_gpu_rank_30_bs64_node66_resnet152.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_30_bs64_node66_resnet152.log-Num devices: 32
log/32_gpu_rank_30_bs64_node66_resnet152.log-Mini batch size [img] : 2048
log/32_gpu_rank_30_bs64_node66_resnet152.log-Time per mini-batch : 0.6958770966529846
log/32_gpu_rank_30_bs64_node66_resnet152.log-Throughput [img/sec] : 2943.0484346308112
log/32_gpu_rank_25_bs64_node66_resnet101.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_25_bs64_node66_resnet101.log-Microbenchmark for network : resnet101
log/32_gpu_rank_25_bs64_node66_resnet101.log---------This process: rank 25--------
log/32_gpu_rank_25_bs64_node66_resnet101.log-Num devices: 1
log/32_gpu_rank_25_bs64_node66_resnet101.log-Mini batch size [img] : 64
log/32_gpu_rank_25_bs64_node66_resnet101.log-Time per mini-batch : 0.48944711208343505
log/32_gpu_rank_25_bs64_node66_resnet101.log-Throughput [img/sec] : 130.75978674707156
log/32_gpu_rank_25_bs64_node66_resnet101.log-
log/32_gpu_rank_25_bs64_node66_resnet101.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_25_bs64_node66_resnet101.log-Num devices: 32
log/32_gpu_rank_25_bs64_node66_resnet101.log-Mini batch size [img] : 2048
log/32_gpu_rank_25_bs64_node66_resnet101.log-Time per mini-batch : 0.48944711208343505
log/32_gpu_rank_25_bs64_node66_resnet101.log-Throughput [img/sec] : 4184.31317590629
log/32_gpu_rank_29_bs64_node66_resnet50.log:--------------------SUMMARY--------------------------
log/32_gpu_rank_29_bs64_node66_resnet50.log-Microbenchmark for network : resnet50
log/32_gpu_rank_29_bs64_node66_resnet50.log---------This process: rank 29--------
log/32_gpu_rank_29_bs64_node66_resnet50.log-Num devices: 1
log/32_gpu_rank_29_bs64_node66_resnet50.log-Mini batch size [img] : 64
log/32_gpu_rank_29_bs64_node66_resnet50.log-Time per mini-batch : 0.2914715671539307
log/32_gpu_rank_29_bs64_node66_resnet50.log-Throughput [img/sec] : 219.5754482158481
log/32_gpu_rank_29_bs64_node66_resnet50.log-
log/32_gpu_rank_29_bs64_node66_resnet50.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
log/32_gpu_rank_29_bs64_node66_resnet50.log-Num devices: 32
log/32_gpu_rank_29_bs64_node66_resnet50.log-Mini batch size [img] : 2048
log/32_gpu_rank_29_bs64_node66_resnet50.log-Time per mini-batch : 0.2914715671539307
log/32_gpu_rank_29_bs64_node66_resnet50.log-Throughput [img/sec] : 7026.414342907139
root@BJ-IDC1-10-10-31-66:/data/pytorch_benchmark# 

