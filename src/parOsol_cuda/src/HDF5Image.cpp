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


#include "Config.h"
#include "HDF5Image.h"
#include <mpi.h>
#include <hdf5.h>
#include <string>
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <omp.h>
#include <sys/time.h>

#include "Timing.h"

void HDF5Image::ReadBC(HDF5_GReader &reader, std::string s,std::vector<unsigned short> & coordinates, std::vector<float> & values)
{
  hsize_t global_dims_of_hdf5[3];
  if (reader.GetSizeOfDataset((s+"_Values").c_str(),global_dims_of_hdf5, 1) > 0)
  {
    std::map<bcitem, double> bcmap;
    double bc_per_cpu = ((double) global_dims_of_hdf5[0])/mpi_size;
    hsize_t my_bc_offset = MyPID*bc_per_cpu;
    hsize_t my_bc_upper_offset =(MyPID+1)*bc_per_cpu; //last element is not included. eg 2  then 0,1 belongs to it
    hsize_t my_bc_count =my_bc_upper_offset - my_bc_offset;
    if (MyPID == mpi_size -1) {
        my_bc_upper_offset = global_dims_of_hdf5[0];
        my_bc_count =my_bc_upper_offset - my_bc_offset;
    }

    float *bcdisp = new float[my_bc_count];
    reader.Read((s+"_Values").c_str(), bcdisp, &my_bc_offset, &my_bc_count, 1);

    unsigned short *bccoords = new unsigned short[my_bc_count*4];
    reader.Read((s+"_Coordinates").c_str(), bccoords, global_dims_of_hdf5[0], my_bc_count, 4, my_bc_offset);
    bcitem tmp;
    unsigned int d =0;
    double disp;
    for(unsigned int i=0; i < my_bc_count*4;) {
        tmp.z = bccoords[i++]; 
        tmp.y = bccoords[i++];
        tmp.x = bccoords[i++];
        tmp.d = bccoords[i++];
        disp = bcdisp[d++];
        if (disp ==0)
            disp = 1e-16;
        bcmap[tmp] = disp;
    }

    for(std::map<bcitem, double>::iterator it =  bcmap.begin(); it != bcmap.end(); ++it) {
        coordinates.push_back(it->first.x);
        coordinates.push_back(it->first.y);
        coordinates.push_back(it->first.z);
        coordinates.push_back(it->first.d);
        values.push_back(it->second);
    }
    delete[] bcdisp;
    delete[] bccoords;
  }
}

HDF5Image::HDF5Image(std::string &fi, CPULayout &layout): _file(fi),_layout(layout), MyPID(0)
{
}

HDF5Image::~HDF5Image()
{
}

int HDF5Image::Scan(BaseGrid* grid)
{
  Timer timer(MPI_COMM_WORLD);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MyPID = mpi_rank;
  PCOUT(MyPID, " Generatin mesh.... \n")

  _procx = _layout.CPUGrid()[0];
  _procy = _layout.CPUGrid()[1];
  _procz = _layout.CPUGrid()[2];

  PCOUT(MyPID, " compute dimension.... using cores in x: "<< _procx << " in y: "<<_procy <<" in z: "<<_procz<<"\n")
  //setting left corner
  grid->corner_x = 0;
  grid->corner_y = 0;
  grid->corner_z = 0;

  HDF5_GReader reader(_file);

  hsize_t global_dims_of_hdf5[3];
  if (!reader.Select("Image_Data")) {
      PCOUT(MyPID, "Error Selecting Image!!!\n")
      MPI_Finalize();
      exit(-1);
  }

  double poisson_ratio = 0.0;
  double voxel_size = 0.0;
  const int invalid_metadata[] = {
    reader.Read("Poisons_ratio", poisson_ratio) != 1 ||
      !(0.0 <= poisson_ratio && poisson_ratio < 0.5),
    reader.Read("Voxelsize", voxel_size) != 1 ||
      !std::isfinite(voxel_size) || voxel_size <= 0.0
  };
  int invalid_on_any_rank[2] = {};
  MPI_Allreduce(invalid_metadata, invalid_on_any_rank, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (invalid_on_any_rank[0]) {
    throw std::invalid_argument("/Image_Data/Poisons_ratio must be a single number in [0, 0.5)");
  }
  if (invalid_on_any_rank[1]) {
    throw std::invalid_argument("/Image_Data/Voxelsize must be a single positive finite number");
  }
  grid->poisons_ratio = poisson_ratio;
  for (int i = 0; i < 3; ++i) {
    grid->res[i] = voxel_size;
  }

  reader.GetSizeOfDataset("Image",global_dims_of_hdf5, 3);

  PCOUT(MyPID, "Image has following dimension: " << global_dims_of_hdf5[0] << " " << global_dims_of_hdf5[1] << " " << global_dims_of_hdf5[2] << std::endl;)

  for(int i=0; i<3; i++) {
    grid->gdim[i] = global_dims_of_hdf5[2-i];
    grid->ldim[i] = global_dims_of_hdf5[2-i];
  }

  grid->_grid =0;


  //Compute the local dimension
  double dims_per_cpu[3];
  for(int i=0; i<3; i++) {
    dims_per_cpu[2-i] = ((double) grid->gdim[i])/_layout.CPUGrid()[i];
  }
  hsize_t my_offset[3] = {};
  hsize_t my_upper_offset[3] ={}; //last element is not included. eg 2  then 0,1 belongs to it
  hsize_t my_count[3] ={};

  for(int i=0;i <3; i++) {
    my_offset[2-i]     = _layout.CPUCoord()[i]*dims_per_cpu[2-i];
    my_upper_offset[2-i]  = (_layout.CPUCoord()[i]+1)*dims_per_cpu[2-i];
    my_count[2-i] = my_upper_offset[2-i] - my_offset[2-i];
  }

  for(int i=0; i <3; i++) {
    if (_layout.CPUCoord()[i] == _layout.CPUGrid()[i] -1) {
      my_upper_offset[2-i] = grid->gdim[i];
      my_count[2-i] = my_upper_offset[2-i] - my_offset[2-i];
    }
  }

  //CPU are xmajor  HDF is zmajor

  for(int i=0; i <3; i++) {
    grid->corner[i] = my_offset[2-i];
    grid->ldim[i] = my_count[2-i];
  }

  timer.Start("Image");
  const hsize_t depth = my_count[0], height = my_count[1], width = my_count[2];
  const hsize_t row_stride = width + 1;
  const hsize_t plane_stride = (height + 1) * row_stride;
  grid->_grid = new double[(depth + 1) * plane_stride]{};  // zero padding on the three upper faces

  // Every MPI rank participates in four reads, including ranks with fewer than four slices.
  constexpr hsize_t reads = 4;
  const hsize_t max_planes = (depth + reads - 1) / reads;
  std::vector<float> slab(std::max<hsize_t>(1, max_planes * height * width));
  hsize_t first_plane = 0;
  for (hsize_t part = 0; part < reads; ++part) {
    my_count[0] = depth / reads + (part < depth % reads);
    reader.Read("Image", slab.data(), my_offset, my_count, 3);
    for (hsize_t z = 0; z < my_count[0]; ++z) {
      for (hsize_t y = 0; y < height; ++y) {
        const hsize_t source = (z * height + y) * width;
        const hsize_t target = (first_plane + z) * plane_stride + y * row_stride;
        std::copy_n(slab.data() + source, width, grid->_grid + target);
      }
    }
    my_offset[0] += my_count[0];
    first_plane += my_count[0];
  }
  timer.Stop("Image");
  t_timing elapsed_time = timer.ElapsedTime("Image");
  PCOUT(MyPID, "Time for Reading the Image: " << COUTTIME(elapsed_time) << "s\n");

  timer.Start("BC");

    timer.Stop("BC");
    ReadBC(reader, std::string("Fixed_Displacement"), grid->fixed_nodes_coordinates, grid->fixed_nodes_values);
    ReadBC(reader, std::string("Loaded_Nodes"), grid->loaded_nodes_coordinates, grid->loaded_nodes_values);
    elapsed_time = timer.ElapsedTime("BC");
    PCOUT(MyPID, "Time for Reading BC: " << COUTTIME(elapsed_time) << "s\n");

   PCOUT(MyPID, "\nHDF5 ImageReader: \n")
   PCOUT(MyPID, "  global Dimension: " << grid->gdim[0] << " " << grid->gdim[1] << " " << grid->gdim[2] << "\n")
   PCOUT(MyPID, "  local Dimension: " << grid->ldim[0] << " " << grid->ldim[1] << " " << grid->ldim[2] << " Resolution: " << grid->res[0] << "\n")
   PCOUT(MyPID, "  Poison's ratio: " << grid->poisons_ratio << "\n")

   long num_gl_bc[2], num_loc_bc[2];
   num_loc_bc[0]=grid->fixed_nodes_values.size();
   num_loc_bc[1]=grid->loaded_nodes_values.size();
   MPI_Reduce(&num_loc_bc, &num_gl_bc, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD );
   PCOUT(MyPID, "  BC: Fixednodesize: " << num_gl_bc[0] << " loadednodesize " << num_gl_bc[1] << "\n")
      return 0;
}
