all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
training:
	./threes --total=100000 --block=1000 --limit=1000 --slide="save=weight.bin alpha=0.003125"
keeptrain:
	./threes --total=100000 --block=1000 --limit=1000 --slide="load=weight.bin alpha=0.003125 save=weight.bin"
stats:
	./threes --total=1000 --slide="load=weights.bin alpha=0" --save="stats.txt"
clean:
	rm threes


for i in {1..10}; do
	./threes --total=100000 --block=1000 --limit=1000 --slide="load=weights.bin save=weights.bin alpha=0.00001" | tee -a train.log
	./threes --total=1000 --slide="load=weights.bin alpha=0" --save="stats.txt"
	tar zcvf weights.$(date +%Y%m%d-%H%M%S).tar.gz weights.bin train.log stats.txt
done