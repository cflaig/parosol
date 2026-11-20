/*
 * PsHash.cpp
 *
 * GParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2013, Daniel Kellenberger
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "PsHash.h"

PsHash::PsHash(std::vector<OctreeNode> &keys,int hashNum,int numHashes) : _maxTriesPerOffsetTableSize(5),_cutOffSearch(50),_binarySearchFrame(100),_hashFactorInc(0.5),_maxPosOffset(65536),_timeSeed(false),_fastConstruction(true) {
	_keys = keys;
	_backSize = 0;
	_hashNum = hashNum;
	_numHashes = numHashes;
	_numKeys = GetNumKeys();
	_domainSize = _keys.back();
	GenerateHashTable();
}

PsHash::PsHash(std::vector<OctreeNode> &keys,int hashNum,int numHashes,bool timeSeed,bool fastConstruction) : _maxTriesPerOffsetTableSize(5),_cutOffSearch(50),_binarySearchFrame(100),_hashFactorInc(0.5),_maxPosOffset(65536),_timeSeed(timeSeed),_fastConstruction(fastConstruction) {
	_keys = keys;
	_backSize = 0;
	_hashNum = hashNum;
	_numHashes = numHashes;
	_numKeys = GetNumKeys();
	_domainSize = _keys.back();
	GenerateHashTable();
}

PsHash::PsHash(std::vector<OctreeNode> &front,std::vector<OctreeNode> &keys,std::vector<OctreeNode> &back,t_index backSize,int hashNum,int numHashes) : _maxTriesPerOffsetTableSize(5),_cutOffSearch(50),_binarySearchFrame(100),_hashFactorInc(0.5),_maxPosOffset(65536),_timeSeed(false),_fastConstruction(true) {
	_front = front;
	_keys = keys;
	_back = back;
	_backSize = backSize;
	_hashNum = hashNum;
	_numHashes = numHashes;
	_numKeys = GetNumKeys();
	if (!_back.empty()) {
		_domainSize = _back.back();
	} else if (!_keys.empty()) {
		_domainSize = _keys.back();
	} else if (!_front.empty()) {
		_domainSize = _front.back();
	} else {
		_domainSize = 0;
	}
	GenerateHashTable();
}

PsHash::PsHash(std::vector<OctreeNode> &front,std::vector<OctreeNode> &keys,std::vector<OctreeNode> &back,t_index backSize,int hashNum,int numHashes,bool timeSeed,bool fastConstruction) : _maxTriesPerOffsetTableSize(5),_cutOffSearch(50),_binarySearchFrame(100),_hashFactorInc(0.5),_maxPosOffset(65536),_timeSeed(timeSeed),_fastConstruction(fastConstruction) {
	_front = front;
	_keys = keys;
	_back = back;
	_backSize = backSize;
	_hashNum = hashNum;
	_numHashes = numHashes;
	_numKeys = GetNumKeys();
	if (!_back.empty()) {
		_domainSize = _back.back();
	} else if (!_keys.empty()) {
		_domainSize = _keys.back();
	} else if (!_front.empty()) {
		_domainSize = _front.back();
	} else {
		_domainSize = 0;
	}
	GenerateHashTable();
}

int PsHash::GetNumKeys() {
	int numKeys = 0;
	std::vector<OctreeNode>::iterator iter;
	for (iter = _front.begin();iter != _front.end();iter++) {
		if (iter->key % _numHashes == _hashNum) {
			numKeys++;
		}
	}
	for (iter = _keys.begin();iter != _keys.end();iter++) {
		if (iter->key % _numHashes == _hashNum) {
			numKeys++;
		}
	}
	uint j = 0;
	for (iter = _back.begin();j < _backSize;iter++,j++) {
		if (iter->key % _numHashes == _hashNum) {
			numKeys++;
		}
	}
	return numKeys;
}

bool PsHash::Test() {
	bool passed = true;
	t_index offset = 0;

	t_octree_key lastKey = -1;
	for (t_index i = 0;i < _front.size();i++,offset++) {
		if (_front[i].key != lastKey) {
			if (_front[i].key % _numHashes == _hashNum) {
				t_index res;
				res = GetValue(_front[i].key);
				if (res != offset) {
					passed = false;
					break;
				}
				lastKey = _front[i].key;
			}
		}
	}

	for (t_index i = 0;i < _keys.size();i++,offset++) {
		if (_keys[i].key % _numHashes == _hashNum) {
			t_index res;
			res = GetValue(_keys[i].key);
			if (res != offset) {
				passed = false;
				break;
			}
		}
	}
	for (t_index i = 0;i < _backSize;i++,offset++) {
		if (_back[i].key % _numHashes == _hashNum) {
			t_index res;
			res = GetValue(_back[i].key);
			if (res != offset) {
				passed = false;
				break;
			}
		}
	}
	if (!passed) {
		std::cout << "hashtest failed\n";
	}
	return passed;
}

void PsHash::GenerateHashTable() {
	bool success = false;
	double hashFactor = 2.0;
	while (!success) {

		_hashTableSize = _numKeys*hashFactor;
		if (_hashTableSize % 2 == 0) {
			_hashTableSize++;
		}
		while (!IsPrime(_hashTableSize)) {
			_hashTableSize += 2;
		}
		_offsetTableSize = 0;
		_maxOffset = std::min(_hashTableSize,(t_index) _maxPosOffset);
		_offsetTableSizeGenerator = new CompactOffsetTableSizeGenerator(_hashTableSize,_hashTableSize + 2,_domainSize,_hashTableSize);
		GenerateOffsetTable(false);
		if (_offsetTableSize > 0) {
			success = true;
		}
		hashFactor += _hashFactorInc;
	}
	if (_fastConstruction) {
		_offsetTableSizeGenerator = new FastOffsetTableSizeGenerator(_numKeys,_domainSize,_hashTableSize);
	} else {
		_offsetTableSizeGenerator = new CompactOffsetTableSizeGenerator(1,_hashTableSize + 6,_domainSize,_hashTableSize);
	}
	_offsetTableSize = 0;
	GenerateOffsetTable(false);
	if (_offsetTableSize == 0) {
		_offsetTableSizeGenerator = new CompactOffsetTableSizeGenerator(_hashTableSize,_hashTableSize + 2,_domainSize,_hashTableSize);
		GenerateOffsetTable(false);
	}
	Test();
}

bool SortDependencies(std::vector<std::pair<t_octree_key,t_index> > i,std::vector<std::pair<t_octree_key,t_index> > j) {
	return i.size() < j.size();
}

void PsHash::GenerateOffsetTable(bool lSuccess) {
	t_index tOffsetTableSize = _offsetTableSizeGenerator->GetSize(lSuccess);
	if (tOffsetTableSize == 0) {
		return;
	}
	std::vector<t_index> tHashTable(_hashTableSize,0);
	std::vector<int> tOffsetTable(tOffsetTableSize,-1);
	//build a dependency list for the offset table, e.g. which keys map to a certain offset table entry
	std::vector<std::vector<std::pair<t_octree_key,t_index> > > dependencies(tOffsetTableSize);
	t_index i = 0;

	t_octree_key lastKey = -1;
	for (std::vector<OctreeNode>::iterator iter = _front.begin();iter != _front.end();iter++) {
		if (iter->key != lastKey && iter->key % _numHashes == _hashNum) {
			std::pair<t_octree_key,t_index> p(iter->key,i);
			dependencies[*iter % tOffsetTableSize].push_back(p);
			lastKey = iter->key;
		}
		i++;
	}
	for (std::vector<OctreeNode>::iterator iter = _keys.begin();iter != _keys.end();iter++) {
		if (iter->key % _numHashes == _hashNum) {
			std::pair<t_octree_key,t_index> p(iter->key,i);
			dependencies[*iter % tOffsetTableSize].push_back(p);
		}
		i++;
	}
	uint bCount = 0;
	for (std::vector<OctreeNode>::iterator iter = _back.begin();bCount < _backSize;iter++,bCount++) {
		if (iter->key % _numHashes == _hashNum) {
			std::pair<t_octree_key,t_index> p(iter->key,i);
			dependencies[*iter % tOffsetTableSize].push_back(p);
		}
		i++;
	}

	//start with the ones with highest dependency count
	sort(dependencies.begin(),dependencies.end(),SortDependencies);
	reverse(dependencies.begin(),dependencies.end());
	while(dependencies.back().empty()) {
		dependencies.pop_back();
	}
	reverse(dependencies.begin(),dependencies.end());
	std::vector<std::vector<std::pair<t_octree_key,t_index> > > tDependencies(dependencies.begin(),dependencies.end());

	bool collision = true;
	int j = 0;
	//try _maxTrieserOffsetTableSize times with different random seed
	while (collision == true && j < _maxTriesPerOffsetTableSize) {
		dependencies.resize(0);
		dependencies.insert(dependencies.begin(),tDependencies.begin(),tDependencies.end());
		std::vector<bool> hashMap(_hashTableSize,false);
		tHashTable.resize(0);
		tHashTable.resize(_hashTableSize,0);
		tOffsetTable.resize(0);
		tOffsetTable.resize(tOffsetTableSize,-1);
		if (_timeSeed) {
			srand(time(NULL));
		} else {
			srand(j);
		}
		collision = false;
		j++;
		while (!dependencies.empty()) {
			std::vector<std::pair<t_octree_key,t_index> > curKeys = dependencies.back();
			dependencies.pop_back();
			std::pair<int,bool> p = GetOffset(curKeys,tOffsetTable,tOffsetTableSize,hashMap);
			if (!p.second) {
				collision = true;
				break;
			}
			t_index offset = p.first;

			tOffsetTable[curKeys.front().first % tOffsetTableSize] = offset;
			for (std::vector<std::pair<t_octree_key,t_index> >::iterator iter = curKeys.begin();iter != curKeys.end();iter++) {
				hashMap[(iter->first + offset) % _hashTableSize] = true;
				tHashTable[(iter->first + offset) % _hashTableSize] = iter->second;
			}
		}
	}
	if (collision) {
		GenerateOffsetTable(false);
	} else {
		_offsetTableSize = tOffsetTableSize;
		_offsetTable.resize(0);
		_offsetTable.insert(_offsetTable.begin(),tOffsetTable.begin(),tOffsetTable.end());
		_hashTable.resize(0);
		_hashTable.insert(_hashTable.begin(),tHashTable.begin(),tHashTable.end());
		if (_fastConstruction) {
			return;
		}
		GenerateOffsetTable(true);
	}
}

//is offset valid for all keys?
bool PsHash::CheckOffset(int offset,std::vector<std::pair<t_octree_key,t_index> >& keys,std::vector<bool>& hashMap) {
	bool collision = false;
	for (std::vector<std::pair<t_octree_key,t_index> >::iterator iter = keys.begin();iter != keys.end();iter++) {
		if (hashMap[(iter->first + offset) % _hashTableSize]) {
			collision = true;
			break;
		}
	}
	return !collision;
}

//find a valid offset for a list keys
std::pair<int,bool> PsHash::GetOffset(std::vector<std::pair<t_octree_key,t_index> >& keys,std::vector<int>& tOffsetTable,t_index tOffsetTableSize,std::vector<bool>& hashMap) {
	t_index key = keys.front().first;
	//try the neighboring entries in the offset table
	int offset[4];
	offset[0] = tOffsetTable[(key + 1) % tOffsetTableSize];
	offset[1] = tOffsetTable[(key + tOffsetTableSize - 1) % tOffsetTableSize];
	for (int i = 0;i < 2;i++) {
		if (offset[i] != -1) {
			if (CheckOffset(offset[i],keys,hashMap)) {
				return std::pair<t_index,bool> (offset[i],true);
			}
		}
	}
	//try the neighboring entries in the hash table
	offset[0] = -1;
	offset[1] = -1;
	offset[2] = -1;
	offset[3] = -1;
	int diffs[2] = {-1,1};
	for (std::vector<std::pair<t_octree_key,t_index> >::iterator iter = keys.begin();iter != keys.end();iter++) {
		for (int i = 0;i < 2;i++) {
			int tOffset = tOffsetTable[(iter->first + tOffsetTableSize + diffs[i]) % tOffsetTableSize];
			if (tOffset != -1) {
				int hOffset = (iter->first + _hashTableSize + diffs[i] + tOffset) % _hashTableSize;
				if (hashMap[hOffset]) {
					for (int j = 0;j < 2;j++) {
						if (!hashMap[(hOffset + diffs[j] + _hashTableSize) % _hashTableSize]) {
							offset[i*2 + j] = (hOffset + diffs[j] - iter->first % _hashTableSize + _hashTableSize) % _hashTableSize;
						} else {
							offset[i*2 + j] = -1;
						}
					}
				}
			}
		}

		for (int i = 0;i < 4;i++) {
			if (offset[i] != -1 && (t_index) offset[i] < _maxOffset) {
				if (CheckOffset(offset[i],keys,hashMap)) {
					return std::pair<t_index,bool> (offset[i],true);
				}
			}
		}
	}

	//try random offset (till no offset is left)
	std::vector<bool> posOffsets(_maxOffset,false);
	t_index i = 0;
	while (i < _maxOffset) {
		offset[0] = rand() % _maxOffset;
		while (posOffsets[offset[0]]) {
			offset[0] = (offset[0] + 1) % _maxOffset;
		}
		posOffsets[offset[0]] = true;
		if (CheckOffset(offset[0],keys,hashMap)) {
			return std::pair<t_index,bool> (offset[0],true);
		}
		i++;
	}
	return std::pair<t_index,bool> (0,false);
}

bool PsHash::IsPrime(t_index n) {
	if (n % 2 == 0 || n % 3 == 0) {
		return false;
	}
	t_index root = (t_index) floor(sqrt(n));
	if (root*root == n) {
		return false;
	}
	for (t_index i = 6;i < root; i++) {
		if (n % (i - 1) == 0 || n % (i + 1) == 0) {
			return false;
		}
	}
	return true;
}
