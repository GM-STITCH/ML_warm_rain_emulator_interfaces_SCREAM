#ifndef TAU_EMULATE_COLLECTION_IMPL_HPP
#define TAU_EMULATE_COLLECTION_IMPL_HPP

#include "p3_functions.hpp" // for ETI only but harmless for GPU
#include "p3_subgrid_variance_scaling_impl.hpp"


// Ensure C linkage for Fortran-callable routine
#ifdef __cplusplus
extern "C" {
#endif
                // global scope. Declare the Fortran subroutine with C linkage
                // This is the C interface to the Fortran subroutine
                void c_tau_emulate_cloud_rain_interactions(const double* qc, const double* nc, const double* qr, const double* nr, 
                                    const double* rho, const double* lamc, const double* lamr, const double* lcldm, const double* n0r, const double* pgam, 
                                    const double* precip_frac, const double* q_small, const int* mgncol, 
                                    double* qc_tend, double* qr_tend, double* nc_tend, double* nr_tend, 
                                    const double* dt);

#ifdef __cplusplus
}
#endif


namespace scream {
namespace p3 {

/*
 * Implementation of TAU bin microphysics emulator.
 * 
 */

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::tau_emulate_collection(
    const Spack& qc_incld, const Spack& nc_incld, const Spack& qr_incld, const Spack& nr_incld, const Spack& rho, const Spack& lamc, const Spack& lamr, const Spack& cld_frac_l, const Spack& logn0r, const Spack& mu_c, const Spack& cld_frac_r, const Scalar& dt, Spack& tau_emulate_qctend, Spack& tau_emulate_nctend, Spack& tau_emulate_qrtend, Spack& tau_emulate_nrtend, const Smask& context)
{

    constexpr Scalar QSMALL = C::QSMALL;    
    //printf("LL: qsmall = %-+.16E\n",QSMALL);   1.0e-14
    //printf("LL: dt = %f\n",dt);  100

    // qsmall to be consistent with KK2000, 1e-8 20250720
    const Scalar QSMALL2 = 1.0e-8;
    const auto qc_large_enough = qc_incld >= QSMALL2 && context;
    //

    const auto qc_not_small = qc_incld >= QSMALL && context;
    const auto qr_not_small = qr_incld >= QSMALL && context;

    const Int mgncol_value = 1;


    // 20250720
    //if(qc_not_small.any() || qr_not_small.any()) {
    if (qc_large_enough.any() || qc_not_small.any() || (qc_not_small.any() && qr_not_small.any()) || qr_not_small.any()) {
    // 20250720
        for(int i = 0; i < Spack::n; ++i) {
            if (qc_not_small[i] || qr_not_small[i]) {
            
                //printf("LL: qc mask = %s\n", qc_not_small[i] ? "true" : "false");
                //printf("LL: qr mask = %s\n", qr_not_small[i] ? "true" : "false");
                //printf("LL: qc_incld, nc_incld, qr_incld, nr_incld = %-+.16E\t %-+.16E\t %-+.16E\t %-+.16E\t%i\n", qc_incld[i], nc_incld[i], qr_incld[i], nr_incld[i], i);
            
                const auto qc_val = qc_incld[i];
                const auto qr_val = qr_incld[i];
                const auto nc_val = nc_incld[i];
                const auto nr_val = nr_incld[i];
    
                const auto rho_val = rho[i];
    
                const auto lamc_val = lamc[i];
                const auto lamr_val = lamr[i];
                const auto n0r_val = pow(10,logn0r[i]);  //un log10, 20250709
                const auto pgam_val = mu_c[i];
       
                const auto lcldm_val = cld_frac_l[i];
                const auto precipfrac_val = cld_frac_r[i];
    
                const auto qsmall_val = QSMALL;
    
                double qctend = 0;
                double qrtend = 0;
                double nctend = 0;
                double nrtend = 0;

                //printf("LL: qc_val, nc_val, qr_val, nr_val = %-+.16E\t %-+.16E\t %-+.16E\t %-+.16E\t%i\n", qc_val, nc_val, qr_val, nr_val, i);

                // C++ code Calling the Fortran subroutine
                c_tau_emulate_cloud_rain_interactions(&qc_val, &nc_val, &qr_val, &nr_val, 
                            &rho_val, &lamc_val, &lamr_val, &lcldm_val, &n0r_val, &pgam_val, 
                            &precipfrac_val, &qsmall_val, &mgncol_value, 
                            &qctend, &qrtend, &nctend, &nrtend, 
                            &dt); // Pass addresses

                //printf("LL: After call qctend = %-+.16E, qrtend = %-+.16E, nctend = %-+.16E, nrtend = %-+.16E\n", qctend, qrtend, nctend, nrtend);

                // 20250714. YPS. Important to care about the sign. be consistent with default model!!
                // from TAU, qctend is negative. qrtend is positive. nctend is negative. nrtend could be negative/positive. This inconsistency once cause NaN values in p3. No error in COMBLE case. NaNs in GoAmazon cases.
                tau_emulate_qctend[i] = (-1)*qctend;
                tau_emulate_nctend[i] = (-1)*nctend;
                tau_emulate_qrtend[i] = qrtend;
                tau_emulate_nrtend[i] = nrtend;

            } else {
                
            tau_emulate_qctend[i] = 0;
            tau_emulate_nctend[i] = 0;
            tau_emulate_qrtend[i] = 0;
            tau_emulate_nrtend[i] = 0;
                
            }
            
        }  // for loop
        
    } // if cond


} //tau_emulate_collection

} // namespace p3
} // namespace scream

#endif
