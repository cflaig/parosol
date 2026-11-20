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
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include "TemporalDisplacementCache.h"

#include <mpi.h>
#include <omp.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>

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
#include "SolveDiagnostics.h"
#include <exception>
#include "GPUMemoryManager.h"
#include "ApplyKernels.h"
#include "../../utilities/cuda_jit_diagnostic.hpp"

#ifdef HASH
	#include "GPUGridBuilderHashSF.h"
#else
	#include "GPUGridBuilderOffsetSF.h"
#endif


#include "VTKPrinter.h"
#include "HDF5Printer.h"

#define EXIT(X) MPI_Finalize(); exit(X);

// Name of the persistent worker's listen socket, relative to its working
// directory (the shared parosol_server directory) -- mirrors the relative-
// path convention already used by the .PAROSOL_IN/.PAROSOL_OUT file
// protocol (see PAROSOL_SERVER_LISTEN_SOCKET in parosol_server.py, which
// this is the C++-side counterpart of, though this is a distinct per-
// worker socket, not the job-queue wake socket).
static const char* WORKER_SOCKET_PATH = ".cudaparosol_worker.sock";

template <class T>
void print(GenericMatrix<T> &matr, Problem<T> &problem, std::string file, int MyPID,GPUMemoryManager *gpuMM);

bool send_line(int fd, const std::string& s);

// Result of a single solve request: everything the caller (either the
// one-shot argv path or the persistent worker loop) needs to report back.
struct JobResult {
	bool success = true;
	std::string error_message;
	double solve_seconds = 0.0;
    std::string diagnostics;
};

// The per-job body that used to be main()'s entire middle section. Builds
// a fresh grid/matrix/preconditioner/solver/problem every call (all local
// stack objects, destroyed at the end of this function -- no per-object
// state survives across calls except the bounded host displacement
// cache below). The two CUDA __constant__ globals declared
// in ApplyKernels.cu (sMat, rpLookupTable) DO persist across calls in a
// persistent-worker context (they are process-lifetime GPU state, not
// per-object state), but both are already written unconditionally by
// constructors that run on every call here (GenericMatrix's ctor calls
// SetLocalStiffnessMatrix; GPUGridBuilderOffsetSF's ctor calls
// SetRpLookupTable) -- there is no lazy-init guard anywhere that would
// skip re-writing them, so each call's own constructors fully overwrite
// whatever a previous call left behind. This is the one correctness
// subtlety a persistent-worker design introduces that a fresh-process-
// per-call design didn't have to worry about; it is exercised directly
// by the two-different-mesh-sizes-back-to-back correctness test, not
// just assumed safe from reading this comment.
JobResult solve_one_job(std::string file, int level, double tol, bool switching,
                        bool temporal_warm_start, SpectralOptions spectral, int MyPID, int progress_fd = -1) {
	JobResult result;

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

	int degree = 10;
	PCOUT(MyPID, "Eigen estimator: " << SpectralModeName(spectral.mode)
	      << " limit=" << spectral.effectiveLimit() << std::endl);
	MlCycle<OctreeKey_Lookup>  prec(grid, matr,&reproTimer,&gridBuilder,&gpuMemoryManager,switching, degree, level-1, 0, 16,0,spectral);
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
    const double cold_residual = resid;
    typedef TemporalDisplacementCache<t_double> DisplacementCache;
    static DisplacementCache displacement_cache;
    // Multi-rank agreement requires a collective cache protocol; this cache is
    // deliberately limited to the supported single-rank CUDA worker.
    const bool warm_enabled = temporal_warm_start && grid.GetNrCPU() == 1;
    if (!warm_enabled) displacement_cache.clear();
    DisplacementCache::Signature signature;
    bool cache_hit = false;
    const double cache_load_start = MPI_Wtime();
    if (warm_enabled) {
        auto& vector = problem.GetSolDev();
        signature.vector_size = vector._syncData.gpuVecSize;
        signature.switching = vector._syncData.switching;
        t_coord dimensions[3]; double spacing[3];
        grid.GetGlobalDim(dimensions); grid.GetRes(spacing);
        for (int k = 0; k < 3; ++k) {
            signature.dimensions[k] = dimensions[k]; signature.spacing[k] = spacing[k];
        }
        signature.nodes.reserve(grid.GetOctGrid().size());
        for (const auto& node : grid.GetOctGrid()) signature.nodes.push_back(node.key);
        const auto& bc = matr.GetBC();
        signature.fixed_indices.assign(bc.FixedNodesInd.begin(), bc.FixedNodesInd.end());
        signature.loaded_indices.assign(bc.LoadedNodesInd.begin(), bc.LoadedNodesInd.end());
        signature.fixed_values = bc.FixedNodes;
        signature.loaded_values = bc.LoadedNodes;
        if (auto* previous = displacement_cache.find(signature)) {
            vector.CopyFromExtCpuVector(previous->displacement.data());
            SetBoundaries3K(matr.GetBC(), vector);
            cache_hit = true;
            resid = problem.Res(); // current K and RHS, never the old system
        }
        solver.SetResidualReference(cold_residual);
    }
    const double cache_load_seconds = MPI_Wtime() - cache_load_start;
    const double initial_residual = resid;
	reproTimer.Reset();
	matTimer.Reset();
	PCOUT(MyPID, "Residuum before solving: " << resid << std::endl);
	timer.Start("Solve");

	if (progress_fd >= 0 && !send_line(progress_fd, "SOLVE_STARTED")) {
		result.success = false;
		result.error_message = "client disconnected before solve started";
		return result;
	}
	const int iterations = problem.Solve();

	resid = problem.Res();
	PCOUT(MyPID, "Residuum after solving: " << resid << std::endl);
	timer.Stop("Solve");
	t_timing solve_time = timer.ElapsedTime("Solve");
	PCOUT(MyPID, "Solving time: " << COUTTIME(solve_time) << "s\n");
	result.solve_seconds = solve_time.avg;
    SolveDiagnostics diagnostics;
    diagnostics.spectral = spectral; diagnostics.backend = "cuda";
    diagnostics.solveScope = "pcg_includes_final_residual_evaluation";
    diagnostics.iterations = iterations; diagnostics.cold = cold_residual;
    diagnostics.initial = initial_residual; diagnostics.final = resid; diagnostics.tolerance = tol;
    diagnostics.preconditionerSeconds = timer.ElapsedTime("Prec").avg;
    diagnostics.solveSeconds = result.solve_seconds;
    diagnostics.warmRequested = temporal_warm_start; diagnostics.warmEnabled = warm_enabled;
    diagnostics.warmHit = cache_hit; diagnostics.cacheLoadSeconds = cache_load_seconds;

    const double cache_save_start = MPI_Wtime();
    bool cache_saved = false;
    // Only converged finite displacements may seed the next request. Preserve
    // the true-residual criterion used to compare the warm and cold arms.
    if (warm_enabled && diagnostics.converged()) {
        std::vector<t_double> displacement(signature.vector_size);
        problem.GetSolDev().CopyToExtCpuVector(displacement.data());
        cache_saved = displacement_cache.store(std::move(signature), std::move(displacement));
    }
    const double cache_save_seconds = MPI_Wtime() - cache_save_start;
    diagnostics.warmSaved = cache_saved; diagnostics.cacheSaveSeconds = cache_save_seconds;
    result.diagnostics = diagnostics.json();
    PCOUT(MyPID, "PAROSOL_RESULT " << result.diagnostics << std::endl);
    if (!diagnostics.converged()) {
        result.success = false;
        result.error_message = std::string("FEM did not converge: ") + diagnostics.reason() + " PAROSOL_RESULT " + result.diagnostics;
        prec.FreeRessources();
        return result;
    }
    PCOUT(MyPID, "TEMPORAL_WARM_START enabled=" << warm_enabled
        << " hit=" << cache_hit << " saved=" << cache_saved
        << " cold_residual=" << cold_residual << " initial_residual=" << initial_residual
        << " final_residual=" << resid << " tolerance=" << tol
        << " iterations=" << iterations << " load_seconds=" << cache_load_seconds
        << " save_seconds=" << cache_save_seconds
        << " host_cache_bytes=" << displacement_cache.bytes() << std::endl);

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

	return result;
}

// Parses the same "--level N --tol T [--switch S] <filename>" grammar the
// one-shot argv path always used, but from a vector of tokens instead of
// argc/argv -- reused by both the one-shot path and the persistent worker,
// which receives one such token line per request over WORKER_SOCKET_PATH
// instead of on the process's own argv (a persistent process only gets one
// argv, at startup).
struct JobRequest {
	bool ok = false;
	int level = 6;
	double tol = 1e-6;
	bool switching = false;
	bool temporal_warm_start = true;
	SpectralOptions spectral;
	std::string error_message;
	std::string file;
};

JobRequest parse_job_tokens(const std::vector<std::string>& tokens) {
	JobRequest req;
    const char* inherited = std::getenv("MICROMPA_CUDA_TEMPORAL_WARM_START");
    bool explicit_warm_start = false;
	bool spectral_mode_seen = false;
	for (size_t k = 0; k < tokens.size(); k++) {
		if (tokens[k] == "--level" && k + 1 < tokens.size()) {
			req.level = atoi(tokens[++k].c_str());
		} else if (tokens[k] == "--tol" && k + 1 < tokens.size()) {
			req.tol = atof(tokens[++k].c_str());
		} else if (tokens[k] == "--switch" && k + 1 < tokens.size()) {
			req.switching = atoi(tokens[++k].c_str());
		} else if (IsSpectralModeOption(tokens[k])) {
			std::string value;
			const std::string option = tokens[k];
                if (option.find('=') == std::string::npos) {
				if (k + 1 >= tokens.size()) { req.error_message = "--eig_estimator_mode requires adaptive or original"; return req; }
				value = tokens[++k];
			} else value = option.substr(option.find('=') + 1);
                if (option.substr(0, option.find('=')) == "--eigen-estimator" && value == "legacy10") value = "original";
			if (spectral_mode_seen || !ParseSpectralMode(value, req.spectral.mode)) {
				req.error_message = "--eig_estimator_mode requires exactly one adaptive or original value"; return req;
			}
			spectral_mode_seen = true;
		} else if (tokens[k] == "--eigen-max-iterations" || tokens[k].rfind("--eigen-max-iterations=", 0) == 0) {
			std::string value;
			if (tokens[k] == "--eigen-max-iterations") {
				if (k + 1 >= tokens.size()) { req.error_message = "--eigen-max-iterations requires an integer from 2 to 1000"; return req; }
				value = tokens[++k];
			} else value = tokens[k].substr(std::string("--eigen-max-iterations=").size());
			if (req.spectral.maxWasSpecified || !ParseSpectralMaximum(value, req.spectral.adaptiveMaxIterations)) {
				req.error_message = "--eigen-max-iterations requires exactly one integer from 2 to 1000"; return req;
			}
			req.spectral.maxWasSpecified = true;
		} else if (tokens[k] == "--temporal-warm-start") {
            if (k + 1 >= tokens.size() || (tokens[k+1] != "0" && tokens[k+1] != "1")) {
                req.error_message = "--temporal-warm-start requires 0 or 1";
                return req;
            }
            req.temporal_warm_start = tokens[++k] == "1";
            explicit_warm_start = true;
		} else {
			req.file = tokens[k];
		}
	}
    if (req.spectral.maxWasSpecified && req.spectral.mode != SpectralMode::Adaptive) {
        req.error_message = "--eigen-max-iterations requires --eig_estimator_mode adaptive";
        return req;
    }
    if (!explicit_warm_start && inherited) {
        const std::string value(inherited);
        if (value != "0" && value != "1") {
            req.error_message = "MICROMPA_CUDA_TEMPORAL_WARM_START requires 0 or 1";
            return req;
        }
        req.temporal_warm_start = value == "1";
    }
	req.ok = !req.file.empty();
    if (!req.ok) req.error_message = "no filename given in request";
	return req;
}

std::vector<std::string> tokenize(const std::string& line) {
	std::istringstream iss(line);
	std::vector<std::string> tokens;
	std::string tok;
	while (iss >> tok) tokens.push_back(tok);
	return tokens;
}

// Minimal blocking line read/write over an already-connected/accepted fd.
// The worker protocol is deliberately simple newline-delimited text (same
// spirit as the argv strings it replaces) -- the actual FE problem data
// still travels entirely through the HDF5 file named in the request line,
// exactly as before; only the small per-job "what to solve" description
// and the "done"/"error" reply move over this socket.
std::string recv_line(int fd) {
	std::string line;
	char c;
	while (true) {
		ssize_t n = read(fd, &c, 1);
		if (n <= 0) break; // EOF or error: caller treats as connection lost
		if (c == '\n') break;
		line += c;
	}
	return line;
}

bool send_line(int fd, const std::string& s) {
	std::string out = s + "\n";
	size_t total = 0;
	while (total < out.size()) {
		ssize_t n = write(fd, out.c_str() + total, out.size() - total);
		if (n <= 0) return false;
		total += static_cast<size_t>(n);
	}
	return true;
}

// Persistent worker mode: pays the MPI_Init/cudaSetDevice cost once (done
// by the caller, main(), before this is entered) and then loops accepting
// one request at a time on socket_path, solving it, and replying, until it
// receives "SHUTDOWN" or the socket is otherwise unusable. Never serves
// more than one request concurrently -- matches the existing
// single-GPU-job-at-a-time serialization already enforced upstream
// (parosol_server's num_parosol_calls gate and the cluster's own GPU QOS
// limits), so no concurrency handling is needed here. socket_path defaults
// to WORKER_SOCKET_PATH but can be overridden (via "--worker --socket
// <path>") so multiple worker processes can coexist on one node without
// colliding on the same socket file.
int run_worker_loop(int MyPID, const std::string& socket_path) {
    int exit_status = 0;
	unlink(socket_path.c_str());

	int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (listen_fd < 0) {
		PCOUT(MyPID, "cudaparosol worker: failed to create socket\n");
		return 2;
	}

	struct sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

	if (bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
		PCOUT(MyPID, "cudaparosol worker: failed to bind " << socket_path << "\n");
		close(listen_fd);
		return 2;
	}
	if (listen(listen_fd, 1) < 0) {
		PCOUT(MyPID, "cudaparosol worker: failed to listen\n");
		close(listen_fd);
		return 2;
	}

	PCOUT(MyPID, "cudaparosol worker ready, listening on " << socket_path << "\n");

	bool run_loop = true;
	while (run_loop) {
		int client_fd = accept(listen_fd, nullptr, nullptr);
		if (client_fd < 0) continue;

		std::string request = recv_line(client_fd);

		if (request.empty()) {
			// A connection that closed without sending anything (e.g. a
			// liveness probe) is not a shutdown request -- just move on
			// to the next accept() instead of tearing down the worker.
			close(client_fd);
			continue;
		}
		if (request == "SHUTDOWN") {
			close(client_fd);
			break;
		}

		JobRequest job = parse_job_tokens(tokenize(request));

		if (!job.ok) {
			send_line(client_fd, "ERROR " + job.error_message);
			close(client_fd);
			continue;
		}

		PCOUT(MyPID, "worker: solving " << job.file << " tolerance: " << job.tol
		             << " max. num. level: " << job.level << std::endl);

		JobResult result;
		try {
			result = solve_one_job(job.file, job.level, job.tol, job.switching,
			                       job.temporal_warm_start, job.spectral, MyPID, client_fd);
		} catch (const std::exception &e) {
			send_line(client_fd, std::string("ERROR Preconditioner setup failed: ") + e.what());
			close(client_fd);
			exit_status = 2;
            break; // A failed setup may have left partial CUDA resources; stop the worker.
		}
		if (!result.success) {
			send_line(client_fd, "ERROR " + result.error_message);
			close(client_fd);
            exit_status = 2;
			break;
		}

		std::ostringstream oss;
		oss << std::setprecision(17) << "OK " << result.solve_seconds << " " << result.diagnostics;
		send_line(client_fd, oss.str());
		close(client_fd);
	}

	close(listen_fd);
	unlink(socket_path.c_str());
	PCOUT(MyPID, "cudaparosol worker shutting down\n");
    return exit_status;
}

int main( int argc, char* argv[]) {

    if (argc == 2 && std::string(argv[1]) == "--supports-spectral-estimator") {
        std::cout << "parosol_spectral_estimator_v2\n";
        return 0;
    }
    // Lightweight build capability check; do not initialize MPI or CUDA.
    if (argc == 2 && std::string(argv[1]) == "--supports-temporal-warm-start") {
        std::cout << "temporal_warm_start_v1\n";
        return 0;
    }

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

	int device_count = 0;
	cudaError_t device_count_err = cudaGetDeviceCount(&device_count);
	if (device_count_err != cudaSuccess || device_count < 1) {
		PCOUT(MyPID, "cudaGetDeviceCount failed: "
		             << cudaGetErrorString(device_count_err) << "\n");
		MPI_Finalize();
		return -2;
	}
	const int device_to_use = (device_count > 1) ? 1 : 0;
	cudaError_t set_device_err = cudaSetDevice(device_to_use);
	if (set_device_err != cudaSuccess) {
		PCOUT(MyPID, "cudaSetDevice(" << device_to_use << ") failed: "
		             << cudaGetErrorString(set_device_err) << "\n");
		MPI_Finalize();
		return -2;
	}
	int dev;
	cudaGetDevice(&dev);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,dev);
	ifb::cuda_diagnostics::log_ptx_jit_requirement_once();

	std::cout << "möp " << deviceProp.name << "\n";

	// --worker: enter persistent worker mode instead of the one-shot
	// argv-driven solve below. MPI/CUDA setup above already happened
	// exactly once; run_worker_loop() then pays no further init cost
	// per request, only per process lifetime. Optional companion
	// "--socket <path>" lets multiple worker processes share one node
	// without colliding on the same socket file (2-slot CUDA streams
	// pool); falls back to the fixed WORKER_SOCKET_PATH default if not
	// given, preserving today's single-worker behavior exactly.
	for (int k = 1; k < argc; k++) {
		if (std::string(argv[k]) == "--worker") {
			std::string socket_path = WORKER_SOCKET_PATH;
			for (int j = 1; j < argc; j++) {
				if (std::string(argv[j]) == "--socket" && j + 1 < argc) {
					socket_path = argv[j + 1];
					break;
				}
			}
			const int status = run_worker_loop(MyPID, socket_path);
			MPI_Finalize();
			return status;
		}
	}

	std::vector<std::string> tokens;
	for (int k = 1; k < argc; k++) tokens.push_back(argv[k]);

	std::string usage = std::string("usage: ") + argv[0]
	                  + std::string(" [--level arg (6)]")
	                  + std::string(" [--tol arg (1e-6)]")
	                  + std::string(" [--eig_estimator_mode adaptive|original (adaptive)] [--eigen-max-iterations 2..1000]")
					  + std::string(" [--switch arg (0)]")
	                  + std::string(" [--temporal-warm-start 0|1 (1)]")
	                  + std::string(" filename\n")
	                  + std::string("   or: ") + argv[0] + std::string(" --worker\n");

	if (argc < 2) {
		PCOUT(MyPID, usage)
		MPI_Finalize();
		return -1;
	}

	JobRequest job = parse_job_tokens(tokens);

	PCOUT(MyPID, "file: " << job.file << " tolerance: " << job.tol << " max. num. level: " << job.level << std::endl);
	if (!job.ok) {
		PCOUT(MyPID, job.error_message << "\n");
		MPI_Finalize();
		return -10;
	}

	try {
		JobResult result = solve_one_job(job.file, job.level, job.tol, job.switching,
		                                 job.temporal_warm_start, job.spectral, MyPID);
		if (!result.success) {
			PCOUT(MyPID, result.error_message << "\n");
			MPI_Finalize(); return 2;
		}
	} catch (const std::exception &e) {
		PCOUT(MyPID, "Preconditioner setup failed: " << e.what() << "\n");
		MPI_Finalize(); return 2;
	}

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

    t_double *sol = problem.GetSol();
    t_double *res = problem.GetRes();
	print.PrintAll(problem.GetSolDev(),sol,tempForce,res);

	timer.Stop("Write");
	t_timing time = timer.ElapsedTime("Write");
	PCOUT(MyPID, "Outputtime: " << COUTTIME(time) << "s\n");
	delete[] tempForce;
	delete dForce;
	delete[] sol;
	delete[] res;
}
