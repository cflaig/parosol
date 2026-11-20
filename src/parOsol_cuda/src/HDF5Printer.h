/*
 * ParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2011, Cyril Flaig
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


#ifndef HDF5PRINTER_H
#define HDF5PRINTER_H

#include "Config.h"
#include "OctreeGrid.h"
#include "Postprocessing.h"
#include "GWriter.hpp"

//! This class prints the grid and result into a HDF5 file.

/*! The printer writes the whole into a single HDF5-file. This file can be read by ParaView.
 *  To read it with ParaView a XML-file is used. This file can be generate with a helper script.
 */

template <class T>
class HDF5Printer {
	public:
		HDF5Printer(std::string filename, OctreeGrid<T> &grid):_MyPID(grid.GetPID()), _Size(grid.GetNrCPU()),_filename(filename), _grid(grid)
		{
		  Writer = new HDF5_GWriter(filename, MPI_COMM_WORLD);
		}
		int _MyPID;
		int _Size;
		~HDF5Printer()
		{
		  Writer->Close();
		  delete Writer;
		}
		
		void OctKey_to_Coord(long key, int &x, int &y, int &z)
		{
			x = y = z = 0;
			for(int i = 0; i < 20; i++) {
				x += (key & 1) << i;
				key = key >> 1;
				y += (key & 1) << i;
				key = key >> 1;
				z += (key & 1) << i;
				key = key >> 1;
			}
		}


		void PrintGrid() {
			PrintCoord("Coordinates");
			PrintElems("Elements");
            PrintEmoduli();
		}
		
		void PrintCoord(std::string dset) {
			Writer->Select("/Mesh");
		  
			t_octree_key k;
			std::vector<OctreeNode> &grid = _grid.GetOctGrid();
			std::vector<OctreeNode>::iterator iter; 
			int x =0,y=0,z=0;
			double res[3];
			_grid.GetRes(res);
            t_double *coord = new t_double[_grid.GetNrPrivateNodes()*3];
			long i=0;
			T keys;
			t_octree_key tmp;
			for(iter = grid.begin(); iter != _grid._GridIteratorEnd; ++iter ) {
				k = iter->key;
				OctKey_to_Coord(k, x, y, z);
				tmp = keys(x,y,z);
				if (tmp != k) {
				  PCOUT(_MyPID, "ERROR not the same key\n")
				  break;
				}
				coord[i++] = x*res[0];
				coord[i++] = y*res[1];
				coord[i++] = z*res[2];
			}
			Writer->Write(dset,coord,_grid.GetNrNodesGlobal(),_grid.GetNrPrivateNodes(),3,_grid.GetNodeOffset());
			delete[] coord;

		}

		void PrintElems(std::string dset) {
			Writer->Select("/Mesh");

			//print element to node
			t_octree_key *elems = new t_octree_key[_grid.GetNrElem()*8];

			t_index local_nodes[8];
			
			//Quick an dirty hack:
			//compute the offset with double
            t_double *ind = new t_double[_grid.GetNrDofs()];
			// Initialize owned IDs before posting the ghost receive.
			for (uint i = 0;i < _grid.GetNrDofs();i++) {
				ind[i] = 0;
			}
			t_octree_key offset = _grid.GetNodeOffset();
			MPI_Barrier(MPI_COMM_WORLD);
			for (t_index i = 0; i < _grid.GetNrPrivateNodes(); i++) {
				ind[3*i] = i + offset;
			}
			MPI_Barrier(MPI_COMM_WORLD);
			_grid.Recv_import_Ghost(ind);
			_grid.Send_import_Ghost(ind);
			_grid.Wait_import_Ghost();
			MPI_Barrier(MPI_COMM_WORLD);

			t_index e =0;
			for(_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()){
				_grid.SearchIndexes(local_nodes); 
				for(int i =0; i <8; i++)
					elems[e*8+i] = (long) ind[3*local_nodes[i]];
				e++;
			}
			MPI_Barrier(MPI_COMM_WORLD);

			Writer->Write(dset,elems,_grid.GetNrElemGlobal(),_grid.GetNrElem(),8,_grid.GetElemOffset());
			delete[] elems;
			delete[] ind;

		}
		
        void PrintEmoduli() {
            Writer->Select("/Mesh");

            t_double *emoduli = new t_double[_grid.GetNrElem()];
			t_index i =0;
			for(_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()){
				emoduli[i]=_grid.GetElementWeight()*1000; 
				i++;
			}
            MPI_Barrier(MPI_COMM_WORLD);
            Writer->Write("Emoduli",emoduli,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());
            delete[] emoduli;

        }

        //x displacement, res residuum
		void PrintAll(DispVector &dispX,t_double *x,t_double *force,t_double *res) {

			PrintGrid();

			PostProcess<OctreeGrid<T> > post(_grid);

			t_double *m = new t_double[_grid.GetNrElem()];
			t_double *s = new t_double[_grid.GetNrElem()];
			t_double *eff = new t_double[_grid.GetNrElem()];
			t_double *e_xx = new t_double[_grid.GetNrElem()];
			t_double *e_yy = new t_double[_grid.GetNrElem()];
			t_double *e_zz = new t_double[_grid.GetNrElem()];
			t_double *e_xy = new t_double[_grid.GetNrElem()];
			t_double *e_yz = new t_double[_grid.GetNrElem()];
			t_double *e_xz = new t_double[_grid.GetNrElem()];

			post.ComputeStressAndStrain(dispX,m,s,eff,e_xx,e_yy,e_zz,e_xy,e_yz,e_xz);
			MPI_Barrier(MPI_COMM_WORLD);

			Writer->Select("/Solution");

			Writer->Write("disp",x,_grid.GetNrNodesGlobal(),_grid.GetNrPrivateNodes(),3,_grid.GetNodeOffset());

			MPI_Barrier(MPI_COMM_WORLD);

			Writer->Write("force", force,_grid.GetNrNodesGlobal(),_grid.GetNrPrivateNodes(),3,_grid.GetNodeOffset());
			MPI_Barrier(MPI_COMM_WORLD);

			Writer->Write("VonMises",m,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());

			MPI_Barrier(MPI_COMM_WORLD);

			Writer->Write("SED",s,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());

			Writer->Write("EFF",eff,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());

			Writer->Write("e_xx",e_xx,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());
			Writer->Write("e_yy",e_yy,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());
			Writer->Write("e_zz",e_zz,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());
			Writer->Write("e_xy",e_xy,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());
			Writer->Write("e_yz",e_yz,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());
			Writer->Write("e_xz",e_xz,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());

			delete[] m;
			delete[] s;
			delete[] eff;
			delete[] e_xx;
			delete[] e_yy;
			delete[] e_zz;
			delete[] e_xy;
			delete[] e_yz;
			delete[] e_xz;

		}

		void PrintPartition(std::string dset) {
			t_double *part = new t_double[_grid.GetNrElem()];
			for (uint i = 0;i < _grid.GetNrElem();i++) {
				part[i] = _MyPID;
			}
			MPI_Barrier(MPI_COMM_WORLD);
			Writer->Write(dset,part,_grid.GetNrElemGlobal(),_grid.GetNrElem(),1,_grid.GetElemOffset());
		}

		std::string _filename;

		OctreeGrid<T> &_grid;
		
		HDF5_GWriter *Writer;

};
#endif /* HDF5PRINTER_H */
