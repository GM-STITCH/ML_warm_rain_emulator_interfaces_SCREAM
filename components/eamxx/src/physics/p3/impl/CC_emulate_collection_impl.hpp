#ifndef CC_EMULATE_COLLECTION_IMPL_HPP
#define CC_EMULATE_COLLECTION_IMPL_HPP

#include "p3_functions.hpp" // for ETI only but harmless for GPU
#include "p3_py_module.hpp" // P3-local handle to the imported py::module

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace scream {
namespace p3 {

/*
 * Implementation of YPS CC microphysics emulator.
 * 
 */

template<typename S, typename D>
void Functions<S,D>
::cc_emulate_collection(
const Spack& qc_incld, const Spack& nc_incld, const Spack& qr_incld, const Spack& nr_incld, const Spack& rho, const Spack& mu_c, const Spack& mu_r, const Scalar& dtimestep, Spack& cc_emulate_qctend, Spack& cc_emulate_nctend, Spack& cc_emulate_qrtend, Spack& cc_emulate_nrtend, const Smask& context)
{

  // Check interpreter state 
  if (!Py_IsInitialized() || g_p3_py_module == nullptr) {
     printf("ALICE: Py is not Init or is Finalizd!\n");
     return;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();

  const auto& py_mod = *g_p3_py_module;

  try { 

       for(int i = 0; i < Spack::n; ++i) {
          
          if (!context[i]) { 
             continue;
          }
     
          const auto qc_val = qc_incld[i] * rho[i] * 1e3;   // unit: g/m3
          const auto qr_val = qr_incld[i] * rho[i] * 1e3;   // unit: g/m3
          const auto nc_val = nc_incld[i] * rho[i] * 1e-6;  // unit: 1/cm3
          const auto nr_val = nr_incld[i] * rho[i] * 1e-6;  // unit: 1/cm3

          // regime 1: include all collection events
          if (qc_val > 1e-2 && qr_val > 1e-2) {

            constexpr int ncol  = 1;
            constexpr int nvars = 6;

            py::array_t<double> input_data({
               static_cast<py::ssize_t>(ncol),
               static_cast<py::ssize_t>(nvars)
            });

            //py::array_t<double> input_data({ncol, nvars});
            auto input = input_data.mutable_unchecked<2>();

            input(0,0) = nc_val;
            input(0,1) = nr_val;
            input(0,2) = qc_val;
            input(0,3) = qr_val;
            input(0,4) = mu_c[i];
            input(0,5) = mu_r[i];

            //printf("ALICE: regime 1!\n"); 
            //printf("%f", nc_val);
     
            // Import YPS CC python module and call function with scalars
            py::array_t<double> result = py_mod.attr("p3_coalescence_emulator")(
                    ncol,nvars,input_data).cast<py::array_t<double>>(); 
 
            // Unpack tuple: retrieve tendency rate terms
            auto prediction = result.unchecked<1>();
            double nctend = prediction(0)/rho[i]/dtimestep;  // unit: 1/kg/s
            double nrtend = prediction(1)/rho[i]/dtimestep;  // unit: 1/kg/s
            double qrtend = prediction(2)/rho[i]/dtimestep;  // unit: kg/kg/s

            //printf("%f", nctend);
            //printf("%f", nrtend);
            //printf("%f", qrtend);

            cc_emulate_qctend[i] = -qrtend;
            cc_emulate_nctend[i] = nctend;
            cc_emulate_qrtend[i] = qrtend;
            cc_emulate_nrtend[i] = nrtend;

          // regime 2: cloud liquid self collection
          } else if (qc_val > 1e-2 && qr_val <= 1e-2) {

            constexpr int ncol  = 1;
            constexpr int nvars = 3;

            py::array_t<double> input_data({
               static_cast<py::ssize_t>(ncol),
               static_cast<py::ssize_t>(nvars)
            });

            //py::array_t<double> input_data({ncol, nvars});
            auto input = input_data.mutable_unchecked<2>();

            input(0,0) = nc_val;
            input(0,1) = qc_val;
            input(0,2) = mu_c[i];

            py::array_t<double> result = py_mod.attr("p3_coalescence_emulator")(
                    ncol,nvars,input_data).cast<py::array_t<double>>();

            // Unpack tuple: retrieve tendency rate terms
            auto prediction = result.unchecked<1>();
            double nctend = prediction(0)/rho[i]/dtimestep;  // unit: 1/kg/s
            double nrtend = prediction(1)/rho[i]/dtimestep;  // unit: 1/kg/s
            double qrtend = prediction(2)/rho[i]/dtimestep;  // unit: kg/kg/s 

            cc_emulate_qctend[i] = -qrtend;
            cc_emulate_nctend[i] = nctend;
            cc_emulate_qrtend[i] = qrtend;
            cc_emulate_nrtend[i] = nrtend;

          // regime 3: rain self collection
          } else if (qc_val <= 1e-2 && qr_val > 1e-2) {

            constexpr int ncol  = 1;
            constexpr int nvars = 2;

            py::array_t<double> input_data({
              static_cast<py::ssize_t>(ncol),
              static_cast<py::ssize_t>(nvars)
            });

            //py::array_t<double> input_data({ncol, nvars});
            auto input = input_data.mutable_unchecked<2>();

            input(0,0) = nr_val;
            input(0,1) = qr_val;

            py::array_t<double> result = py_mod.attr("p3_coalescence_emulator")(
                    ncol,nvars,input_data).cast<py::array_t<double>>();

            // Unpack tuple: retrieve tendency rate terms
            auto prediction = result.unchecked<1>();
            double nctend = 0.0;  // unit: 1/kg/s
            double nrtend = prediction(1)/rho[i]/dtimestep;  // unit: 1/kg/s
            double qrtend = 0.0;  // unit: kg/kg/s

            cc_emulate_qctend[i] = -qrtend; 
            cc_emulate_nctend[i] = nctend;
            cc_emulate_qrtend[i] = qrtend;
            cc_emulate_nrtend[i] = nrtend;

          }  else {

                cc_emulate_qctend[i] = 0;
                cc_emulate_nctend[i] = 0;
                cc_emulate_qrtend[i] = 0;
                cc_emulate_nrtend[i] = 0;

            }

        }  // for loop

  } catch (py::error_already_set& e) {
           printf("ALICE: Python call error: %s\n",  e.what());
           PyGILState_Release(gstate);
           throw;
         }

  PyGILState_Release(gstate);

} //cc_emulate_collection

} // namespace p3
} // namespace scream

#endif
