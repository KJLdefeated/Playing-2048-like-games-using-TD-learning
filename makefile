all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
training:
	./threes --total=100000 --block=1000 --limit=1000 --slide="save=weight.bin"
keeptrain:
	./threes --total=100000 --block=1000 --limit=1000 --slide="load=weight.bin alpha=0.0025 save=weight.bin" | tee -a train14.log
stats:
	./threes --total=1000 --slide="load=weight.bin alpha=0" --save="stats.txt"
clean:
	rm threes