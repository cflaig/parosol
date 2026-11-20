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

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "Config.h"
#include "StiffnessMatrix.h"
#include "Toolbox.h"
#include "fem.h"
#include "ApplyKernels.h"


/*! The PostProcess Class computes the von mises stress and the strain energy density
 *  from the resulting displacement of the computation.
 */


template <class Grid>
class PostProcess 
{
	public:
		PostProcess(Grid &grid) :_grid(grid)
	{
	}

		//! ElementByElementMatrix destructor
		~PostProcess() { }

		void ComputeStressAndStrain(DispVector &disp,t_double *VonMises,t_double *SED,t_double *eff) {
			//fetch the nodes of the neighbours
			disp.PrepareIn(0);
			_grid.Recv_import_Ghost(disp._syncData);
			_grid.Send_import_Ghost(disp._syncData);

			// Set the reference element up
			double GridDim[3];
			double *coord = new double[24];
			_grid.GetRes(GridDim);
			setcoord(GridDim,coord);

			t_index nr_elem =_grid.GetNrElem();
			for (uint i = 0;i < nr_elem;i++) {
				VonMises[i] = 0.0;
				SED[i] = 0.0;
				eff[i] = 0.0;
			}


			//fetch the 24 values in pref and store store
			// res = K_e * xpref

			_grid.Wait_import_Ghost();
			disp.PostProcess(0);
			MPI_Barrier(MPI_COMM_WORLD);


			DispVector *dVec = new DispVectorN(disp._syncData,disp._gpuMM);

			t_double *dev_coord;
			CCS(cudaMalloc((void**) &dev_coord,24*sizeof(t_double)),"tesfa");
			CCS(cudaMemcpy(dev_coord,coord,24*sizeof(t_double),cudaMemcpyHostToDevice),"safew");

			_grid._gpuGrid->PostProcessK(disp.GetVector(),dVec->GetVector(),dev_coord,_grid.poisons_ratio);

			t_double *res = new t_double[disp._syncData.gpuVecSize];
			dVec->CopyToExtCpuVector(res);

			int c = 0;
			int r = 0;
			for (std::vector<OctreeNode>::iterator iter = _grid._OctreeGrid.begin();iter != _grid._GridIteratorEnd;iter++) {
				if (iter->w >= 0) {
					VonMises[r] = res[c*3];
					SED[r] = res[c*3 + 1];
					eff[r] = sqrt(2*SED[r]/(1000*iter->w));
					r++;
				}
				c++;
			}

			delete dVec;
			delete res;
			delete coord;
			CCS(cudaFree(dev_coord),"dfas");
		}

		void ComputeStressAndStrain(t_double *disp,t_double *VonMises,t_double *SED,t_double *eff) {
			//fetch the nodes of the neighbours
			_grid.Recv_import_Ghost(disp);
			_grid.Send_import_Ghost(disp);

            // Set the reference element up
			double GridDim[3];
			int Dimension = 3;
			const int NumMaterialProps = 2;
			double _matprop[2];
			_matprop[0] = 1000; //reference value Emodule is linear
			_matprop[1] = _grid.poisons_ratio;
			int NumNodesPerElement = 8;
			int NumDofsPerElement = 24;
			int NumGaussPoints = 1;
			int SSMatrixSize = 6;
			double *coord = new double[Dimension * NumNodesPerElement];
			_grid.GetRes(GridDim);
			setcoord(GridDim,coord);

			double* strainbuf = new double[(SSMatrixSize +1) * NumGaussPoints ];
			double* stressbuf = new double[(SSMatrixSize +1) * NumGaussPoints ];
			double sigma, theta;
			t_index nr_elem =_grid.GetNrElem();
			for (uint i = 0;i < nr_elem;i++) {
				VonMises[i] = 0.0;
				SED[i] = 0.0;
				eff[i] = 0.0;
			}


			//fetch the 24 values in pref and store store
			// res = K_e * xpref
			t_double *xpref = new t_double[24];
			int nr_ele = 0;

			_grid.Wait_import_Ghost();
			MPI_Barrier(MPI_COMM_WORLD);

			for(_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()){

				_grid.GetNodalDisplacementsOfElement(disp,xpref);

				_matprop[0] = 1000*_grid.GetElementWeight();
                double emoduli = 1000*_grid.GetElementWeight();
				Element_Stress(_matprop, NumMaterialProps,
						NumNodesPerElement, NumDofsPerElement,
	                    Dimension, NumGaussPoints, SSMatrixSize,
						coord, &xpref[0], 
						strainbuf, stressbuf, 
						&sigma, &theta);
				VonMises[nr_ele] = stressbuf[6];
				SED[nr_ele] = strainbuf[6];
                eff[nr_ele] = sqrt(2*SED[nr_ele]/emoduli); 

				nr_ele++;

			}
			delete[] strainbuf;
			delete[] stressbuf;
			delete[] xpref;
		}

	private:
		Grid& _grid;
};

#endif /* POSTPROCESSING_H */
