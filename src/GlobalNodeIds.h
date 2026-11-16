#ifndef PAROSOL_GLOBAL_NODE_IDS_H
#define PAROSOL_GLOBAL_NODE_IDS_H

#include <Eigen/Core>

// Global, one-based output indices in the first component of each node.
// Finish initialization before posting receives: a neighbouring rank may
// already have sent its IDs and MPI may immediately fill the ghost slots.
template <class Grid>
void SetGlobalNodeIds(Grid &grid, Eigen::VectorXd &ids) {
  ids.setZero(grid.GetNrDofs());
  const auto offset = grid.GetNodeOffset();
  for (decltype(grid.GetNrPrivateNodes()) i = 0;
       i < grid.GetNrPrivateNodes(); ++i)
    ids[3 * i] = i + offset + 1;
  grid.Recv_import_Ghost(ids);
  grid.Send_import_Ghost(ids);
  grid.Wait_import_Ghost();
}

#endif
