/*
 * PsHash.h
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

#ifndef PSHASH_H_
#define PSHASH_H_

#include <vector>
#include <utility>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "Config.h"
#include "FastOffsetTableSizeGenerator.h"
#include "CompactOffsetTableSizeGenerator.h"
#include "OctreeNode.h"

class PsHash {

public:
	PsHash(std::vector<OctreeNode> &keys,int hashNum,int numHashes);
	PsHash(std::vector<OctreeNode> &keys,int hashNum,int numHashes,bool timeSeed,bool fastConstruction);
	PsHash(std::vector<OctreeNode> &front,std::vector<OctreeNode> &keys,std::vector<OctreeNode> &back,t_index backSize,int hashNum,int numHashes);
	PsHash(std::vector<OctreeNode> &front,std::vector<OctreeNode> &keys,std::vector<OctreeNode> &back,t_index backSize,int hashNum,int numHashes,bool timeSeed,bool fastConstruction);

	~PsHash() {

	}

	std::vector<t_index>& GetHashTable() {
		return _hashTable;
	}

	t_index GetHashTableSize() {
		return _hashTableSize;
	}

	std::vector<t_offset>& GetOffsetTable() {
		return _offsetTable;
	}

	t_index GetOffsetTableSize() {
		return _offsetTableSize;
	}

	t_index GetValue(t_octree_key key) {
		return GetIndexOfKey(key);
	}

	void SetOffset(t_octree_key key,t_index offset) {
		_hashTable[(key + _offsetTable[key % _offsetTableSize]) % _hashTableSize] = offset;
	}

private:

	std::vector<OctreeNode> _front;
	std::vector<OctreeNode> _keys;
	std::vector<OctreeNode> _back;
	t_index _backSize;
	t_index _numKeys;
	t_index _maxOffset;
	t_index _domainSize;
	t_index _hashTableSize;
	t_index _offsetTableSize;
	std::vector<t_index> _hashTable;
	std::vector<t_offset> _offsetTable;

	int _hashNum;
	int _numHashes;

	AOffsetTableSizeGenerator *_offsetTableSizeGenerator;

	const int _maxTriesPerOffsetTableSize;
	const int _cutOffSearch;
	const int _binarySearchFrame;
	const double _hashFactorInc;
	const int _maxPosOffset;
	const bool _timeSeed;
	const bool _fastConstruction;

	bool Test();
	void GenerateHashTable();
	void GenerateOffsetTable(bool lSuccess);
	int GetNumKeys();
	bool CheckOffset(int offset,std::vector<std::pair<t_octree_key,t_index> >& keys,std::vector<bool>& hashMap);
	std::pair<int,bool> GetOffset(std::vector<std::pair<t_octree_key,t_index> >& keys,std::vector<int>& tOffsetTable,t_index tOffsetTableSize,std::vector<bool>& hashMap);
	bool IsPrime(t_index n);

	inline t_index GetIndexOfKey(t_octree_key key) {
		return _hashTable[(key + _offsetTable[key % _offsetTableSize]) % _hashTableSize];
	}
};

#endif /* PSHASH_H_ */
