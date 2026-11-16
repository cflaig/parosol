#include "est_ev.h"
#include "Chebyshev.h"
#include <Eigen/Eigenvalues>
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool ok,const char *message) {
  if(!ok)throw std::runtime_error(message);
}

// Replicated operator with disjoint ownership for MPI inner products.
class Matrix : public StiffnessMatrix {
 public:
  Eigen::MatrixXd a;
  Eigen::VectorXd inverseDiagonal;
  int rank,ranks,calls=0;
  explicit Matrix(Eigen::MatrixXd matrix):a(std::move(matrix)) {
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    inverseDiagonal=a.diagonal().cwiseInverse();
  }
  int Apply(Eigen::VectorXd &x,Eigen::VectorXd &y) override {++calls;y=a*x;return 0;}
  const std::string Label() const override {return "test matrix";}
  Eigen::VectorXd &Diagonal() override {return inverseDiagonal;}
  t_index GetNrDofs() override {return a.rows();}
  double dot(Eigen::VectorXd &x,Eigen::VectorXd &y) override {
    const int begin=a.rows()*rank/ranks,end=a.rows()*(rank+1)/ranks;
    double local=x.segment(begin,end-begin).dot(y.segment(begin,end-begin)),sum;
    MPI_Allreduce(&local,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);return sum;
  }
  int GetPID() override {return rank;}
  void SetVectorRandom(Eigen::VectorXd &x) override {
    x.resize(a.rows());for(int i=0;i<a.rows();++i)x[i]=std::sin((i+1)*1.234567);
  }
};
class Identity : public Solver {
 public:
  int Solve(Eigen::VectorXd &b,Eigen::VectorXd &x) override {x=b;return 0;}
  const std::string Label() const override {return "identity";}
};
class ZeroPreconditioner : public Solver {
 public:
  int Solve(Eigen::VectorXd &b,Eigen::VectorXd &x) override {x.setZero(b.size());return 0;}
  const std::string Label() const override {return "zero preconditioner";}
};

int main(int argc,char **argv) {
 MPI_Init(&argc,&argv);
 int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 try {
  Identity identity;Eigen::VectorXd r,d;
  Matrix single(4*Eigen::MatrixXd::Identity(32,32));
  eig exact=est_ev(single,identity,100,r,d);
  require(exact.converged && exact.invariantSubspace && exact.iterations==1,
          "one-eigenvalue operator must terminate cleanly in one step");
  require(std::abs(exact.large-4)<1e-12 && std::abs(exact.small-4)<1e-12,"incorrect exact spectrum");
  require(single.calls==exact.iterations,"convergence checks added matrix products");

  Eigen::MatrixXd a=Eigen::MatrixXd::Zero(160,160);
  for(int i=0;i<a.rows();++i)a(i,i)=1+9.*i/(a.rows()-1);
  Matrix spread(a);
  const eig limited=est_ev(spread,identity,2,r,d);
  require(!limited.converged && limited.iterations==2,"short cap must report nonconvergence");
  const eig adaptive=est_ev(spread,identity,100,r,d);
  require(adaptive.converged && adaptive.iterations<100,"smooth spectrum did not converge adaptively");
  require(adaptive.large>9.8 && adaptive.large<=10+1e-10,"incorrect largest eigenvalue");
  require(adaptive.relativeResidual<0.01 && adaptive.relativeChange<0.01,"acceptance thresholds violated");
  const eig repeat=est_ev(spread,identity,100,r,d);
  require(repeat.iterations==adaptive.iterations && std::abs(repeat.large-adaptive.large)<1e-12,
          "estimate is not reproducible");

  // Largest eigenvector (1,-1,0,...) is invisible to the old constant start.
  a(0,0)=a(1,1)=500.5;a(0,1)=a(1,0)=-499.5;
  Matrix hidden(a);
  const eig found=est_ev(hidden,identity,100,r,d);
  require(found.converged && std::abs(found.large-1000)<1e-5,"missed largest mode orthogonal to constants");

  Matrix zero(Eigen::MatrixXd::Zero(8,8));
  bool rejected=false;
  try {est_ev(zero,identity,20,r,d);}catch(const std::runtime_error &){rejected=true;}
  require(rejected,"zero operator must not return uninitialized eigenvalues");
  rejected=false;
  try {est_ev(spread,identity,0,r,d);}catch(const std::invalid_argument &){rejected=true;}
  require(rejected,"zero iteration limit must be rejected");

  // Original regression: a zero preconditioned starting norm skipped every
  // iteration and invoked the tridiagonal solver with n=0, writing at -1.
  ZeroPreconditioner zeroPreconditioner;
  const int before=spread.calls;
  rejected=false;
  try {est_ev(spread,zeroPreconditioner,20,r,d);}catch(const std::runtime_error &){rejected=true;}
  require(rejected && spread.calls==before,"zero starting norm must fail before matrix application");
  Matrix empty(Eigen::MatrixXd::Zero(0,0));
  rejected=false;
  try {est_ev(empty,identity,20,r,d);}catch(const std::runtime_error &){rejected=true;}
  require(rejected && empty.calls==0,"globally empty system must fail before matrix application");

  // A limited, unconverged estimate must never silently initialize Chebyshev.
  Matrix block(a);Eigen::VectorXd w1(a.rows()),w2(a.rows());
  rejected=false;
  try {Chebyshev smoother(block,6,true,w1,w2,2);}catch(const std::runtime_error &){rejected=true;}
  require(rejected,"Chebyshev accepted an unconverged spectral interval");
  Chebyshev smoother(block,6,true,w1,w2,100);
  if(rank==0)std::cout<<"PASS adaptive spectrum: exact="<<exact.iterations
      <<" smooth="<<adaptive.iterations<<" hidden_mode="<<found.iterations<<'\n';
 }catch(const std::exception &e) {
  std::cerr<<"rank "<<rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;
 }
 MPI_Finalize();return 0;
}
