/*
 * CompactOffsetTableSizeGenerator.cpp
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

#include "CompactOffsetTableSizeGenerator.h"

t_index CompactOffsetTableSizeGenerator::GetSize(bool lSuccess) {
	if (lSuccess) {
		_oneSuccess = true;
	}
	t_index res;
	if (_lastSize != 0) {
		if (lSuccess) {
			_uBound = _lastSize;
		} else {
			_lBound = _lastSize;
		}
	}
	if (_lBound + 1 == _uBound) {
		return 0;
	}
	res = _lBound + (_uBound - _lBound)/2;
	if (_uBound - _lBound < _cutOffSearch && _oneSuccess) {
		return 0;
	}
	bool smallFrame = _uBound - _lBound < _binarySearchFrame;
	for (t_index i = res;i > _lBound;i--) {
		if (_domainSize > i*_hashTableSize) {
			break;
		}
		if (GCD(i,_hashTableSize) != 1) {
			continue;
		}
		if (!smallFrame && (_hashTableSize % i == 1 || _hashTableSize % i == i - 1)) {
			continue;
		}
		_lastSize = i;
		return i;
	}
	for (t_index i = res;i < _uBound;i++) {
		if (_domainSize > i*_hashTableSize || GCD(i,_hashTableSize)) {
			continue;
		}
		if (!smallFrame && (_hashTableSize % i == 1 || _hashTableSize % i == i - 1)) {
			continue;
		}
		_lastSize = i;
		return i;
	}
	return 0;
}
