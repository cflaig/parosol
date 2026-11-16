#include "ImageReader.h"
#include "OctreeGrid.h"
#include "GenericMatrix.h"
#include "KeyGenerator.h"
#include <cmath>

double global_poisson_ratio = 0.3;
using Base = OctreeGrid<OctreeKey_Lookup>;

class Image : public ImageReader {
public:
  explicit Image(int size) : n(size) {}
  int Scan(BaseGrid *g) override {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int start = n * rank / ranks, end = n * (rank + 1) / ranks;
    g->gdim_x = g->gdim_y = g->gdim_z = n;
    g->ldim_x = g->ldim_y = n; g->ldim_z = end - start;
    g->corner_x = g->corner_y = 0; g->corner_z = start;
    g->res_x = g->res_y = g->res_z = 1.0 / n;
    g->_grid = new double[n * n * (end - start)];
    for (int z = start; z < end; ++z)
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
          g->_grid[((z - start) * n + y) * n + x] =
              (x < 4 ? 0.01 : 1.0) * (1 + (x + 2*y + 3*z) % 7);
    if (rank == 0)
      for (int y = 0; y <= n; ++y)
        for (int x = 0; x <= n; ++x)
          for (int c = 0; c < 3; ++c) {
            for (int v : {x, y, 0, c}) g->fixed_nodes_coordinates.push_back(v);
            g->fixed_nodes_values.push_back(0);
          }
    return 0;
  }
private:
  int n;
};

// Exercise the production matrix with real MPI communication. Completing an
// import immediately makes an overwrite after Recv deterministic, instead of
// relying on a particular MPI progress engine or message arrival time.
class Grid : public Base {
public:
  explicit Grid(Image &image) : Base(image) {}
  bool completeImmediately = false;
  void Recv_import_Ghost(Eigen::VectorXd &v) {
    Base::Recv_import_Ghost(v);
    if (completeImmediately) {
      Base::Send_import_Ghost(v);
      Base::Wait_import_Ghost();
    }
  }
  void Send_import_Ghost(Eigen::VectorXd &v) {
    if (!completeImmediately) Base::Send_import_Ghost(v);
  }
  void Wait_import_Ghost() {
    if (!completeImmediately) Base::Wait_import_Ghost();
  }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  {
    const int n = argc == 2 && std::string(argv[1]) == "8" ? 8 : 12;
    Image image(n);
    Grid grid(image);
    grid.GenerateBC();
    if (grid.GetNrElemGlobal() != n*n*n || grid.GetNrNodesGlobal() != (n+1)*(n+1)*(n+1))
      MPI_Abort(MPI_COMM_WORLD, 5);
    if (n == 8 && grid.GetNrCPU() == 4) {
      // The fourth partition receives only helper nodes during Sort().
      // It must not acquire a fabricated key-zero node or any owned DOFs.
      if (grid.GetPID() == 3 && grid.GetNrNodes() != 0)
        MPI_Abort(MPI_COMM_WORLD, 6);
    }
    using Matrix = GenericMatrix<Grid>;
    Matrix reference(grid);
    Eigen::VectorXd expected = Eigen::VectorXd::Zero(grid.GetNrDofs());
    Eigen::VectorXd probe(grid.GetNrDofs()), applied(grid.GetNrDofs());
    // Same-parity distinct Q1 nodes never share an element. These 24 probes
    // recover the diagonal independently of GenericMatrix::Diagonal().
    for (int color = 0; color < 8; ++color)
      for (int c = 0; c < 3; ++c) {
        probe.setZero();
        for (unsigned i = 0; i < grid.GetNrPrivateNodes(); ++i)
          if ((grid.GetOctGrid()[i].key & 7) == color) probe[3*i+c] = 1;
        reference.Apply(probe, applied);
        for (unsigned i = 0; i < grid.GetNrPrivateNodes(); ++i)
          if ((grid.GetOctGrid()[i].key & 7) == color)
            expected[3*i+c] = applied[3*i+c] == 0 ? 0 : 1 / applied[3*i+c];
      }
    grid.Recv_import_Ghost(expected);
    grid.Send_import_Ghost(expected);
    grid.Wait_import_Ghost();

    unsigned long long ghosts = grid.GetNrNodes() - grid.GetNrPrivateNodes();
    unsigned long long totalGhosts = 0;
    MPI_Allreduce(&ghosts, &totalGhosts, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (grid.GetNrCPU() > 1 && totalGhosts == 0) MPI_Abort(MPI_COMM_WORLD, 1);
    for (bool immediate : {false, true}) {
      grid.completeImmediately = immediate;
      Matrix matrix(grid);
      const auto &actual = matrix.Diagonal();
      double error = 0;
      // Include ghosts: checking only owned entries misses this regression.
      for (int i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) MPI_Abort(MPI_COMM_WORLD, 2);
        error = std::max(error, std::abs(actual[i] - expected[i]) / (1 + std::abs(expected[i])));
      }
      double globalError;
      MPI_Allreduce(&error, &globalError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (grid.GetPID() == 0)
        std::cout << "Diagonal: immediate=" << immediate << " error=" << globalError << '\n';
      if (globalError > 2e-12) MPI_Abort(MPI_COMM_WORLD, 3);
      if (&matrix.Diagonal() != &actual) MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (n == 8) {
      // Also exercise empty receive buffers and partitions containing only
      // helper nodes directly, without requiring a particular load balance.
      grid.GetOctGrid().clear();
      grid.Sort();
      if (!grid.GetOctGrid().empty()) MPI_Abort(MPI_COMM_WORLD, 7);
      grid.GetOctGrid().emplace_back(100 + grid.GetPID(), -2);
      grid.Sort();
      if (!grid.GetOctGrid().empty()) MPI_Abort(MPI_COMM_WORLD, 8);
      if (grid.GetPID() == 0)
        std::cout << "Empty and helper-only partitions remain empty\n";
    }
  }
  MPI_Finalize();
}
