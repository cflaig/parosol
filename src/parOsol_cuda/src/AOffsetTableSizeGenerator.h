/*
 * AOffsetTableSizeGenerator.h
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

#ifndef AOFFSETTABLESIZEGENERATOR_H_
#define AOFFSETTABLESIZEGENERATOR_H_

#include "Config.h"

class AOffsetTableSizeGenerator {

public:

	AOffsetTableSizeGenerator(t_index domainSize,t_index hashTableSize) : _domainSize(domainSize),_hashTableSize(hashTableSize) {

	}

	virtual ~AOffsetTableSizeGenerator() {

	}

	virtual t_index GetSize(bool lSuccess) = 0;
	t_index GetSize();

protected:

	t_index _lastSize;
	const t_index _domainSize;
	const t_index _hashTableSize;

	t_index GCD(t_index a,t_index b);

};

#endif /* AOFFSETTABLESIZEGENERATOR_H_ */
