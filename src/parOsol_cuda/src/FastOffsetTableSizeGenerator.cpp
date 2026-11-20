/*
 * FastOffsetTableSizeGenerator.cpp
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

#include "FastOffsetTableSizeGenerator.h"

t_index FastOffsetTableSizeGenerator::GetSize(bool lSuccess) {
	t_index res;
	if (_lastSize == 0) {
		res = (t_index) ceil(_numElements*_startFactor);
	} else {
		res = _lastSize + (int) _lastInc;
		_lastInc = _lastInc*_cRatio;
	}
	while (true) {
		if (res > _hashTableSize) {
			return 0;
		}
		if (_domainSize < res*_hashTableSize && IsPrime(res) && res != _hashTableSize && _hashTableSize % res != 1 && _hashTableSize % res != res - 1) {
			_lastSize = res;
			return res;
		}
		res++;
	}
}

bool FastOffsetTableSizeGenerator::IsPrime(t_index n) {
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
