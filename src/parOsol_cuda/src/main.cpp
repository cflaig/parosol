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

#include <iostream>

#include <mpi.h>
#include <omp.h>

#include "Timing.h"
#include "Config.h"

template <class T> class OctreeGrid;
template <class T> class MlOctreeGrid;

//is chosen by the templates
#include "KeyGenerator.h"

#include "HDF5Image.h"
#include "OctreeGrid.h"
#include "GenericMatrix.h"
#include "PCGSolver.h"
#include "PCGSolverMC.h"
#include "Problem.h"
#include "MlCycle.h"
#include "GPUMemoryManager.h"
#include "ApplyKernels.h"

#ifdef HASH
	#include "GPUGridBuilderHashSF.h"
#else
	#include "GPUGridBuilderOffsetSF.h"
#endif


#include "VTKPrinter.h"
#include "HDF5Printer.h"

#define EXIT(X) MPI_Finalize(); exit(X);

template <class T>
void print(GenericMatrix<T> &matr, Problem<T> &problem, std::string file, int MyPID,GPUMemoryManager *gpuMM);

int main( int argc, char* argv[]) {

	int pid,psize,MyPID;
	int provided;

	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
	MPI_Comm_rank (MPI_COMM_WORLD, &pid);
	MyPID = pid;
	MPI_Comm_size (MPI_COMM_WORLD, &psize);

	if (provided < MPI_THREAD_FUNNELED) {
		PCOUT(MyPID,"only MPI_THREAD_SINGLE supported\n");
	}

	int _numThreads;
#pragma omp parallel
	if (omp_get_thread_num() == 0) {
		_numThreads = omp_get_num_threads();
	}

	PCOUT(MyPID,_numThreads << " threads are available.\n");

	cudaSetDevice(1);
	int dev;
	cudaGetDevice(&dev);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,dev);

	std::cout << "möp " << deviceProp.name << "\n";

	std::string file;
    int level = 6, degree = 10;
    double tol = 1e-6;
    bool switching = false;

    std::string usage = std::string("usage: ") + argv[0]
                      + std::string(" [--level arg (6)]")
                      + std::string(" [--tol arg (1e-6)]")
    				  + std::string(" [--switch arg (0)]")
                      + std::string(" filename\n"); 
	
    if (argc < 2) {
		PCOUT(MyPID, usage)
		MPI_Finalize();
		return -1;
	}
	
    int i = 1;
    std::string param;
    while(i < argc) {
        param = std::string(argv[i]);
        if (param.compare("--level") == 0) {
            i++;
            level = atoi(argv[i]);
        } else if (param.compare("--tol") == 0) {
            i++;
            tol = atof(argv[i]);
        } else if (param.compare("--switch") == 0) {
        	i++;
        	switching = atoi(argv[i]);
        } else {
            file = param;
        }
        i++;
    }

	PCOUT(MyPID, "file: " << file << " tolerance: " << tol << " max. num. level: " << level << std::endl);
    if (file.compare("") == 0) {
		PCOUT(MyPID, "No file given\n");
		MPI_Finalize();
		return -10;
    }


	Timer timer(MPI_COMM_WORLD);
	timer.Start("All");

	t_timing elapsed_time;

	CPULayout layout;

	GPUMemoryManager gpuMemoryManager;

#ifdef HASH
	GPUGridBuilderHashSF<OctreeKey_Lookup> gridBuilder(&gpuMemoryManager);
#else
	GPUGridBuilderOffsetSF<OctreeKey_Lookup> gridBuilder(&gpuMemoryManager);
#endif

	Timer reproTimer(MPI_COMM_WORLD);

    //Read in the Problem
	timer.Start("Read");
	HDF5Image ir(file, layout);
	typedef OctreeGrid<OctreeKey_Lookup> t_Ogrid;
	t_Ogrid grid(ir,&gridBuilder,&reproTimer,level);
	
	timer.Stop("Read");
	elapsed_time = timer.ElapsedTime("Read");
	PCOUT(MyPID, "Time for Read in: "  << COUTTIME(elapsed_time) << "s\n");

	PCOUT(MyPID, "Nr. of global elements: " << grid.GetNrElemGlobal() << " global nodes: " << grid.GetNrNodesGlobal() << "\n");

    //Generate the BC
	timer.Start("BC");
	grid.GenerateBC();
	timer.Stop("BC");
	elapsed_time = timer.ElapsedTime("BC");
	PCOUT(MyPID, "Time for Construction of Boundary Condition: " << COUTTIME(elapsed_time) << "s\n");

	PCOUT(MyPID, grid);

    //Construct the matrix
	timer.Start("Mat");
	Timer matTimer(MPI_COMM_WORLD);
	GenericMatrix<t_Ogrid> matr(grid,level,level,&matTimer);
	matr.SetGPUMemoryManager(&gpuMemoryManager);
	timer.Stop("Mat");
	elapsed_time = timer.ElapsedTime("Mat");
	PCOUT(MyPID, "Time for Construction of Matrix: " << COUTTIME(elapsed_time) << "s\n");
	PCOUT(MyPID, matr);

	std::cout << std::setprecision (13); 
	MPI_Barrier(MPI_COMM_WORLD);

    //Construct the preconditioner
	timer.Start("Prec");

	MlCycle<OctreeKey_Lookup>  prec(grid, matr,&reproTimer,&gridBuilder,&gpuMemoryManager,switching, degree, level-1, 0, 16,0);
	PCOUT(MyPID, "Prec done\n");
	timer.Stop("Prec");
	elapsed_time = timer.ElapsedTime("Prec");
	PCOUT(MyPID, "Time for Building Preconditioner: " << COUTTIME(elapsed_time) << "s\n");

    //Setting up the problem
	timer.Start("Setup");
    int maxiter = 1000;
    int outputfreq = 4;

	PCGSolverMC solver(matr,prec,prec.GetEFine(),prec.GetRFine(),tol,maxiter,true,outputfreq);

	solver.SetGPUMemoryManager(&gpuMemoryManager);
	Problem<t_Ogrid> problem(matr,&gpuMemoryManager);
	if (problem.Impose())
		std::cout << "Error in impose\n";
	problem.SetSolver(solver);
	timer.Stop("Setup");
	elapsed_time =  timer.ElapsedTime("Setup");
	PCOUT(MyPID, "Time for Setting up the Problem: " << COUTTIME(elapsed_time) << "s\n");

    //solving the system
	double resid = problem.Res();
	reproTimer.Reset();
	matTimer.Reset();
	PCOUT(MyPID, "Residuum before solving: " << resid << std::endl);
	timer.Start("Solve");

	problem.Solve();

	resid = problem.Res();
	PCOUT(MyPID, "Residuum after solving: " << resid << std::endl);
	timer.Stop("Solve");
	t_timing time = timer.ElapsedTime("Solve");
	PCOUT(MyPID, "Solving time: " << COUTTIME(time) << "s\n");

	//gpuMemoryManager.PrintDistribution(MyPID);

	prec.FreeRessources();
    //print out the solution
	//prec.PrintTimings();
    print(matr, problem, file, MyPID,&gpuMemoryManager);

	timer.Stop("All");
	elapsed_time = timer.ElapsedTime("All");

	PCOUT(MyPID, "Overall time: " << COUTTIME(elapsed_time) << "s\n");

	long maxIn = gpuMemoryManager.GetMax();
	long long max = 0;
	MPI_Allreduce(&maxIn,&max,1,MPI_LONG_LONG,MPI_MAX,MPI_COMM_WORLD);

	PCOUT(MyPID,"Max memory usage (per node): " << max/1024/1024 << " MB\n");

	//mflops(matr, MyPID);

	MPI_Finalize();
	return 0;
}

template <class T>
void print(GenericMatrix<T> &matr, Problem<T> &problem, std::string file, int MyPID,GPUMemoryManager *gpuMM) {
	Timer timer(MPI_COMM_WORLD);
	//VTKPrinter<OctreeKey_Lookup> print("out", matr.GetGrid());
	HDF5Printer<OctreeKey_Lookup> print(file, matr.GetGrid());
	//matr.PrintTimings();
	timer.Start("Write");

    t_double *tempForce = new t_double[problem.GetSolDev()._syncData.gpuVecSize];
    DispVector *dForce;
    if (matr.GetSyncData().switching) {
    	dForce = new DispVectorS(matr.GetSyncData(),gpuMM);
    } else {
    	dForce = new DispVectorN(matr.GetSyncData(),gpuMM);
    }
    matr.Apply_NoResetBoundaries(problem.GetSolDev(),*dForce);
    dForce->CopyToExtCpuVector(tempForce);

	print.PrintAll(problem.GetSolDev(),problem.GetSol(),tempForce,problem.GetRes());

	timer.Stop("Write");
	t_timing time = timer.ElapsedTime("Write");
	PCOUT(MyPID, "Outputtime: " << COUTTIME(time) << "s\n");
	delete tempForce;
}

