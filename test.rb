
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


