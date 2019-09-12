
file_name = ARGV[0]


#log/4_gpu_rank_3_bs64_node66_vgg16.log:--------------------SUMMARY--------------------------
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Microbenchmark for network : vgg16
#log/4_gpu_rank_3_bs64_node66_vgg16.log---------This process: rank 3--------
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Num devices: 1
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Mini batch size [img] : 64
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Time per mini-batch : 0.6132083201408386
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Throughput [img/sec] : 104.3690992080812
#log/4_gpu_rank_3_bs64_node66_vgg16.log-
#log/4_gpu_rank_3_bs64_node66_vgg16.log---------Overall (all ranks) (assuming same num/type devices for each rank)--------
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Num devices: 4
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Mini batch size [img] : 256
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Time per mini-batch : 0.6132083201408386
#log/4_gpu_rank_3_bs64_node66_vgg16.log-Throughput [img/sec] : 417.4763968323248
#

result = Hash.new
# key: model, batch_size, gpu_num, rank_id, throughput 

f = File.open(file_name, "r")


f.read.split("SUMMARY").each do |s|

  if s =~ /Overall/
  summary = s.split("Overall")[0]
  overall = s.split("Overall")[1]

  #puts s
  #puts summary
  #puts overall
  #exit


  model = ''
  batch_size = ''
  gpu_num = ''
  rank_id = ''
  throughput = ''

  summary.split("\n").each do |l|
    if l =~ /for network/
      model = l.split(":")[1].gsub(/\s*/,'')
    elsif l =~ /This process/
      rank_id = l.scan(/rank\s+(\d+)/).join
    elsif l =~ /batch size/
      batch_size = l.split(":")[1].gsub(/\s*/,'')
    elsif l =~ /Throughput/
      throughput = (l.split(":")[1].gsub(/\s*/,'')).to_f.round(2)
    end
  end

  overall.split("\n").each do |l|
    if l =~ /Num devices/
      gpu_num = l.split(":")[1].gsub(/\s*/,'')
    end
  end

  puts "#{model},#{batch_size},#{gpu_num},#{rank_id},#{throughput}"


  end # if s=~ /Overall/
end
