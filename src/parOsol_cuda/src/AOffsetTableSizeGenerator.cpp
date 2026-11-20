/*
 * AOffsetTableSizeGenerator.cpp
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

#include "AOffsetTableSizeGenerator.h"

t_index AOffsetTableSizeGenerator::GetSize() {
	return GetSize(false);
}

//Compute greatest common divisor
t_index AOffsetTableSizeGenerator::GCD(t_index a,t_index b) {
	t_index ta = a;
	t_index tb = b;
		while (ta != tb) {
		if (ta > tb) {
			ta -= tb;
		} else {
			tb -= ta;
		}
	}
	return ta;
}
