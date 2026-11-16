#include "GlobalNodeIds.h"
#include <iostream>
#include <stdexcept>

// Two owned nodes followed by one ghost. Model both permitted receive timings:
// data already available when Recv is posted, and completion only during Wait.
class Grid {
public:
  explicit Grid(bool early) : early_(early) {}
  unsigned GetNrPrivateNodes() const { return 2; }
  unsigned GetNrDofs() const { return 9; }
  unsigned long GetNodeOffset() const { return 10; }
  void Recv_import_Ghost(Eigen::VectorXd &ids) {
    if (ids[0] != 11 || ids[3] != 12 || ids[6] != 0)
      throw std::runtime_error("receive posted before initialization completed");
    ids_ = &ids;
    if (early_) receive();
  }
  void Send_import_Ghost(Eigen::VectorXd &ids) {
    if (&ids != ids_ || ids[0] != 11 || ids[3] != 12)
      throw std::runtime_error("incorrect outgoing node IDs");
    if (early_ && ids[6] != 101)
      throw std::runtime_error("completed receive was overwritten");
    sent_ = true;
  }
  void Wait_import_Ghost() {
    if (!sent_) throw std::runtime_error("wait called before send");
    if (!early_) receive();
  }
private:
  void receive() { (*ids_)[6] = 101; }
  bool early_, sent_ = false;
  Eigen::VectorXd *ids_ = nullptr;
};

int main() {
  try {
    for (bool early : {false, true}) {
      Grid grid(early);
      Eigen::VectorXd ids = Eigen::VectorXd::Constant(9, -999);
      SetGlobalNodeIds(grid, ids);
      for (int i = 0; i < 9; ++i) {
        const double expected = i == 0 ? 11 : i == 3 ? 12 : i == 6 ? 101 : 0;
        if (ids[i] != expected) throw std::runtime_error("wrong global node ID");
      }
    }
    std::cout << "early and delayed ghost receives preserve global node IDs\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
