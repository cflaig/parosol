/*
 * CompactOffsetTableSizeGenerator.h
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

#ifndef COMPACTOFFSETTABLESIZEGENERATOR_H_
#define COMPACTOFFSETTABLESIZEGENERATOR_H_

#include "AOffsetTableSizeGenerator.h"

class CompactOffsetTableSizeGenerator : public AOffsetTableSizeGenerator {

public:

	CompactOffsetTableSizeGenerator(t_index lBound,t_index uBound,t_index domainSize,t_index hashTableSize) : AOffsetTableSizeGenerator(domainSize,hashTableSize),_cutOffSearch(50),_binarySearchFrame(100) {
		_lBound = lBound;
		_uBound = uBound;
		_lastSize = 0;
		_oneSuccess = false;
	}

	t_index GetSize(bool lSuccess);

private:
	t_index _lBound;
	t_index _uBound;
	bool _oneSuccess;
	const unsigned int _cutOffSearch;
	const unsigned int _binarySearchFrame;

};

#endif /* COMPACTOFFSETTABLESIZEGENERATOR_H_ */
