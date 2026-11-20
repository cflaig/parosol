/*
 * FastOffsetTableSizeGenerator.h
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

#ifndef FASTOFFSETTABLESIZEGENERATOR_H_
#define FASTOFFSETTABLESIZEGENERATOR_H_

#include "AOffsetTableSizeGenerator.h"
#include "math.h"

class FastOffsetTableSizeGenerator : public AOffsetTableSizeGenerator {

public:

	FastOffsetTableSizeGenerator(t_index numElements,t_index domainSize,t_index hashTableSize) : AOffsetTableSizeGenerator(domainSize,hashTableSize),_startFactor(0.5),_cRatio(3),_numElements(numElements) {
		_lastSize = 0;
		_lastInc = 1.0;
	}

	t_index GetSize(bool lSuccess);

private:

	double _lastInc;
	const double _startFactor;
	const double _cRatio;
	const t_index _numElements;
	bool IsPrime(t_index n);

};

#endif /* FASTOFFSETTABLESIZEGENERATOR_H_ */
