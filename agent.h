/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"
#include "weight.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
protected:
	static const int MAX_INDEX = 16;
	static const int TUPLE_NUM = 4;
	static const int TUPLE_LEN = 6;
	std::vector<weight> net;
	static const std::array<std::array<int, TUPLE_LEN>, TUPLE_NUM> indexs;
	float alpha;
	
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0.025f) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		else init_network();
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		// 8x4: 16^4 =  ; 8x4x6: 16^6 = 16777216
		std::string res = info;
		for (char& ch : res) if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}

	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

	float board_value(const board&b){
		float ans = 0;
		board tmp = b;
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++)
				ans += net[i][get_feature(tmp, indexs[i])];
		}
		tmp.reflect_horizontal();
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++)
				ans += net[i][get_feature(tmp, indexs[i])];
		}
		tmp.reflect_horizontal();
		tmp.reflect_vertical();
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++)
				ans += net[i][get_feature(tmp, indexs[i])];
		}
		return ans;
	}

	int get_feature(const board &b, const std::array<int, TUPLE_LEN> index){
		int ans=0;
		int tile;
		for(auto i:index){
			ans *= MAX_INDEX;
			tile = b[i/4][i%4];
			if(tile >= MAX_INDEX - 1)ans += (MAX_INDEX-1);
			else ans += tile; 
		}
		return ans;
	}

	virtual void init_network(){
		int feature_num = MAX_INDEX * MAX_INDEX * MAX_INDEX * MAX_INDEX * MAX_INDEX * MAX_INDEX;
		for(int i=0;i<TUPLE_NUM;i++){
			net.push_back(weight(feature_num));
		}
	}
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

class greedy_slider : public agent {
public:
	virtual action take_action(const board& before) {
		int best_r=-1, best_a=-1, act[4]={0,1,2,3};
		int p = board(before).slide(act[0]);
		if(p > best_r){
			best_r = p;
			best_a = act[0];
		}
		p = board(before).slide(act[1]);
		if(p > best_r){
			best_r = p;
			best_a = act[1];
		}
		if(best_r!=-1)return action::slide(best_a);
		p = board(before).slide(act[2]);
		if(p > best_r){
			best_r = p;
			best_a = act[2];
		}
		p = board(before).slide(act[3]);
		if(p > best_r){
			best_r = p;
			best_a = act[3];
		}
		if(best_r!=-1)return action::slide(best_a);
		return action();
	}

};

class weight_slider : public weight_agent {
protected:
	struct state{
		board st;
		int rew;
	};
	std::vector<state> episode;
public:

	weight_slider(const std::string& args) : weight_agent("name=slide role=slider " + args){}

	virtual void open_episode(const std::string& flag = "") {
		episode.clear();
	}

	virtual void close_episode(const std::string& flag = "") {
		train_weight(episode[episode.size()-1].st);
		for(int i=episode.size()-2;i>=0;i--){
			state ept = episode[i+1];
			train_weight(episode[i].st, ept.st, ept.rew);
		}
	}

	virtual action take_action(const board& before){
		int best_op = 0;
		float best_val = -9999999;
		board b_t;
		int best_rew = 0;

		for(int op:{0,1,2,3}){
			board b = before;
			int rew = b.slide(op);
			if(rew != -1){
				float val = rew + board_value(b);
				if(val > best_val){
					best_val = val;
					best_op = op;
					b_t = b;
					best_rew = rew;
				}
			}
		}

		if(best_val != -9999999){
			episode.push_back(state{b_t, best_rew});
		}

		return action::slide(best_op);
	}

	void change_lr(){
		std::cout << "Learning Rate: " << alpha;
		alpha/=2;
		std::cout << " change to: " << alpha << '\n';
	}

	//train weight for the last episode
	void train_weight(const board& b){
		float delta = -alpha * board_value(b);
		board tmp = b;
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++){
				net[i][get_feature(tmp, indexs[i])] += delta;
			}
		}
		tmp.reflect_horizontal();
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++){
				net[i][get_feature(tmp, indexs[i])] += delta;
			}
		}
		tmp.reflect_horizontal();
		tmp.reflect_vertical();
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++){
				net[i][get_feature(tmp, indexs[i])] += delta;
			}
		}
	}
	//train weight for the episode+1 and episode
	void train_weight(const board& b, const board& b_t, const int rew){
		float delta = alpha * (rew + board_value(b_t) - board_value(b));
		board tmp = b;
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++){
				net[i][get_feature(tmp, indexs[i])] += delta;
			}
		}
		tmp.reflect_horizontal();
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++){
				net[i][get_feature(tmp, indexs[i])] += delta;
			}
		}
		tmp.reflect_horizontal();
		tmp.reflect_vertical();
		for(int j=0;j<4;j++){
			tmp.rotate();
			for(int i=0;i<TUPLE_NUM;i++){
				net[i][get_feature(tmp, indexs[i])] += delta;
			}
		}
	}
};

//The index of features in N-tuple 8x4x6
const std::array<std::array<int, weight_agent::TUPLE_LEN>, weight_agent::TUPLE_NUM> weight_agent::indexs = {{
	{{0, 4, 8, 9, 12, 13}},
	{{1, 5, 9, 10, 13, 14}},
	{{1, 2, 5, 6, 9, 10}},
	{{2, 3, 6, 7, 10, 11}}
}};