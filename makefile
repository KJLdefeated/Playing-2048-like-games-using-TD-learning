all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
stats:
	./threes --total=1000 --save=stats.txt
initw_stats:
	weights_size="65536,65536,65536,65536,65536,65536,65536,65536"
	./threes --total=100000 --block=1000 --limit=1000 --slide="init=$weights_size save=weights.bin"
snapshots_stats:
	weights_size="65536,65536,65536,65536,65536,65536,65536,65536"
	./threes --total=0 --slide="init=$weights_size save=weights.bin"
	for i in {1..100}; do
		./threes --total=100000 --block=1000 --limit=1000 --slide="load=weights.bin save=weights.bin alpha=0.0025" | tee -a train.log
		./threes --total=1000 --slide="load=$weights.bin alpha=0" --save="stats.txt"
		tar zcvf weights.$(date +%Y%m%d-%H%M%S).tar.gz weights.bin train.log stats.txt
	done
load_from_file:
	./threes --total=1000 --slide="load=weights.bin alpha=0" --save="stats.txt" # need to inherit from weight_agent
clean:
	rm threes
