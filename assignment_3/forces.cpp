#include <eigen3/Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// This is crucial. Without 'Eigen::RowMajor', data will be scrambled.
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// 2. The Physics Kernel
void compute_harmonic_forces(py::array_t<double> pos_array,
                             py::array_t<double> force_array, double k,
                             double d0) {

  // --- THE MAGIC: EIGEN MAPS ---

  // Get info about the arrays (pointers, dimensions)
  py::buffer_info pos_info = pos_array.request();
  py::buffer_info force_info = force_array.request();

  // Check dimensions (Safety First!)
  if (pos_info.ndim != 2 || pos_info.shape[1] != 3) {
    throw std::runtime_error("Positions must be (N, 3) array");
  }

  int n_beads = pos_info.shape[0];

  // Wrap the raw C++ pointers from NumPy into Eigen Objects
  // No copying happens here. 'positions' is just a "lens" over the NumPy
  // memory.
  Eigen::Map<RowMatrixXd> positions(static_cast<double *>(pos_info.ptr),
                                    n_beads, 3);
  Eigen::Map<RowMatrixXd> forces(static_cast<double *>(force_info.ptr), n_beads,
                                 3);

  // --- THE PHYSICS LOOP (Vectorized in Eigen) ---

  forces.setZero(); // Clear forces

  // Loop through bonds (linear chain)
  for (int i = 0; i < n_beads - 1; i++) {
    // High-level vector math on raw memory
    // Vector pointing from i to i+1
    Eigen::Vector3d r_vec = positions.row(i + 1) - positions.row(i);

    double dist = r_vec.norm();

    // Avoid division by zero
    if (dist < 1e-6)
      continue;

    // Scalar force magnitude (Hooke's law: f = -k(d-d0))
    double f_mag = -k * (dist - d0);

    // Convert scalar force to vector force
    Eigen::Vector3d f_vec = f_mag * (r_vec / dist);

    // Apply Newton's 3rd Law
    forces.row(i + 1) += f_vec; // Force on i+1
    forces.row(i) -= f_vec;     // Equal and opposite force on i
  }
}

// 3. The Pybind11 Module Definition
PYBIND11_MODULE(polymer_engine, m) {
  m.doc() = "C++ backend for Polymer Simulation"; // Optional module docstring
  m.def("compute_harmonic_forces", &compute_harmonic_forces,
        "Compute spring forces");
}
