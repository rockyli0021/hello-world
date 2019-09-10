# hello-world
hello world

this is a hello world test readme file


https://raw.githubusercontent.com/wiki/ROCmSoftwarePlatform/pytorch/shufflenet_v2.py

find -name '4gpu*.log' -exec grep -A15 SUMM {} \;
--------------------SUMMARY--------------------------
Microbenchmark for network : vgg16
--------This process: rank 1--------
Num devices: 1
Mini batch size [img] : 64
Time per mini-batch : 0.5805431207021078
Throughput [img/sec] : 110.24159570196701

--------Overall (all ranks) (assuming same num/type devices for each rank)--------
Num devices: 4
Mini batch size [img] : 256
Time per mini-batch : 0.5805431207021078
Throughput [img/sec] : 440.96638280786806
--------------------SUMMARY--------------------------
Microbenchmark for network : vgg16
--------This process: rank 0--------
Num devices: 1
Mini batch size [img] : 64
Time per mini-batch : 0.5417807102203369
Throughput [img/sec] : 118.12897504965031

--------Overall (all ranks) (assuming same num/type devices for each rank)--------
Num devices: 4
Mini batch size [img] : 256
Time per mini-batch : 0.5417807102203369
Throughput [img/sec] : 472.51590019860123
--------------------SUMMARY--------------------------
Microbenchmark for network : vgg16
--------This process: rank 3--------
Num devices: 1
Mini batch size [img] : 64
Time per mini-batch : 0.5362457434336344
Throughput [img/sec] : 119.34826669243412

--------Overall (all ranks) (assuming same num/type devices for each rank)--------
Num devices: 4
Mini batch size [img] : 256
Time per mini-batch : 0.5362457434336344
Throughput [img/sec] : 477.3930667697365
--------------------SUMMARY--------------------------
Microbenchmark for network : vgg16
--------This process: rank 2--------
Num devices: 1
Mini batch size [img] : 64
Time per mini-batch : 0.49493153889973956
Throughput [img/sec] : 129.3108136577345

--------Overall (all ranks) (assuming same num/type devices for each rank)--------
Num devices: 4
Mini batch size [img] : 256
Time per mini-batch : 0.49493153889973956
Throughput [img/sec] : 517.243254630938





find -name '4gpu*.log' -exec grep -A15 SUMM {} \;
--------------------SUMMARY--------------------------
Microbenchmark for network : vgg16
--------This process: rank 1--------
Num devices: 1
Mini batch size [img] : 64
Time per mini-batch : 0.5805431207021078
Throughput [img/sec] : 110.24159570196701

--------Overall (all ranks) (assuming same num/type devices for each rank)--------
Num devices: 4
Mini batch size [img] : 256
Time per mini-batch : 0.5805431207021078
Throughput [img/sec] : 440.96638280786806
--------------------SUMMARY--------------------------
Microbenchmark for network : vgg16
--------This process: rank 0--------
Num devices: 1
Mini batch size [img] : 64
Time per mini-batch : 0.5417807102203369
Throughput [img/sec] : 118.12897504965031

--------Overall (all ranks) (assuming same num/type devices for each rank)--------
Num devices: 4
Mini batch size [img] : 256
Time per mini-batch : 0.5417807102203369
Throughput [img/sec] : 472.51590019860123
--------------------SUMMARY--------------------------
Microbenchmark for network : vgg16
--------This process: rank 3--------
Num devices: 1
Mini batch size [img] : 64
Time per mini-batch : 0.5362457434336344
Throughput [img/sec] : 119.34826669243412

--------Overall (all ranks) (assuming same num/type devices for each rank)--------
Num devices: 4
Mini batch size [img] : 256
Time per mini-batch : 0.5362457434336344
Throughput [img/sec] : 477.3930667697365
--------------------SUMMARY--------------------------
Microbenchmark for network : vgg16
--------This process: rank 2--------
Num devices: 1
Mini batch size [img] : 64
Time per mini-batch : 0.49493153889973956
Throughput [img/sec] : 129.3108136577345

--------Overall (all ranks) (assuming same num/type devices for each rank)--------
Num devices: 4
Mini batch size [img] : 256
Time per mini-batch : 0.49493153889973956
Throughput [img/sec] : 517.243254630938
