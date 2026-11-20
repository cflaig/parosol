/*
 * GParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2011, Cyril Flaig
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


#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <utility>
#include <sys/time.h>

#define NOFINEKEYS UINT_MAX
#define NUMHASH 16

//#define COMPUTE35
//#define HASH


//#define MMX  //use the mmx-intrinsics in case of switching

#define COUT std::cout
#define COUTPRECISION(pres) std::cout.precision(pres)

/* 32bit is enough for the index */
/// type of the local array index
typedef unsigned int t_index;

/// type of the coordinate
typedef unsigned int t_coord;

/// type of the octree key
typedef long t_octree_key;

/// type of offset table
typedef unsigned short t_offset;

typedef double t_double;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

struct OffsetCpu {
	t_index index[8];
	t_double weight;
};

struct HashMaskGpu {
	t_index *dev_hashTable;
	t_index hashTableSize;
	t_index hashTableMod;
	t_offset *dev_offsetTable;
	t_index offsetTableSize;
	t_index offsetTableMod;
};

struct HashMask {
	t_octree_key maskx;
	t_octree_key masky;
	t_octree_key maskz;
	t_octree_key *dev_keys;
	t_double *dev_weights;
	HashMask *hashMaskFine;
	HashMaskGpu *dev_hashMaskGpu;
};

union offsetD {
	ulong asuLong;
	uint asuInt[2];
	ushort asuShort[4];
	uchar asuChar[8];
};

struct ElementOffsets {
	t_index *dev_startInds;
	offsetD *dev_offsets_long1;
	offsetD *dev_offsets_long2;
	offsetD *dev_offsets_long3;
	t_index *dev_offsets_last;
	t_double *dev_weights;
	t_index breakPoint0;
	t_index breakPoint1;
	t_index breakPoint2;
	t_index *dev_fineStartInds;
	uchar *dev_localInds;
};

union Off8 {
	uchar keys[4];
	uint offsetInt;
	Off8() {

	}
	Off8(uchar x,uchar y,uchar z,uchar w) {
		keys[0] = x;
		keys[1] = y;
		keys[2] = z;
		keys[3] = w;
	}
};

union Offset8 {
	struct {
		t_index start;
		uchar keys[8];
	} offsetKeys;
	uint offsetInt[3];
};

union Offset16 {
	struct {
		t_index start;
		ushort keys[8];
	} offsetKeys;
	uint offsetInt[5];
};

union Offset32 {
	struct {
		t_index start;
		uint keys[7];
	} offsetKeys;
	uint offsetInt[8];
};
/// holds direction and displacement of the BC
struct boundary_disp
{
	boundary_disp(short dir=0, float disp=0):dir(dir), disp(disp)
	{
	}

	short dir;
	float disp;
	bool operator < (const boundary_disp &rhs) const {
		return dir < rhs.dir;
	}
};

/// A boundary node is consist of an node and bc properties
typedef std::pair<t_octree_key, boundary_disp> t_boundary_node;

//some Marcros

#define COUT_RANK0 std::cout
//cout only on cpunode 0
#define PCOUT(PID, msg) \
	    if (PID == 0) COUT_RANK0 << msg;

//cout on all cpus with barrier 
#define ALL_OUT(PID, SIZE, msg) \
{ \
    int i=0; \
    MPI_Status status; \
    MPI_Barrier(MPI_COMM_WORLD); \
    if ( 0 != PID) { \
        MPI_Recv(&i, 1, MPI_INT, PID -1,0, MPI_COMM_WORLD, &status); \
    } \
 \
    COUT << msg; \
    COUT.flush(); \
    if (PID != SIZE -1) { \
        MPI_Send(&i, 1, MPI_INT, PID+1, 0, MPI_COMM_WORLD);  \
    } \
    MPI_Barrier(MPI_COMM_WORLD); \
}

//cout a cuda error string if available and add the given string
#define CCS(error,str) { \
	if (error != cudaSuccess) { \
		std::cout << cudaGetErrorString(cudaGetLastError()) << " | " << str << std::endl; \
	} \
} \

//cout a cuda error string if available
#define CC(error) { \
	if (error != cudaSuccess) { \
		std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl; \
	} \
} \

#endif /*CONFIG_H*/
