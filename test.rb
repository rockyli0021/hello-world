
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



