#pragma once

// -----------------------------------------------------------------------------
// LesEwaldKokkosT: latent Ewald (LES) monopole energy and analytic forces for a
// periodic triclinic cell.
//
// Reference implementation (formula-for-formula port):
//   chengucb-les/src/les/module/ewald.py::Ewald.compute_potential_triclinic
//
// Physics / units
//   * positions in Angstrom, charges in elementary charge e, energy in eV.
//   * C is the 3x3 cell matrix, ROW-major, with lattice vectors on its ROWS:
//         C = [ a1 ]        volume V = |det C|
//             [ a2 ]
//             [ a3 ]
//     This matches the host convention used elsewhere in the engine
//     (cell[3*row + column], e.g. MACEKokkos::update_factorized_cell and the
//     polar k-space setup in mace_polar_impl.tpp).
//   * G = 2*pi*inv(C)^T, k = n @ G for integer n.
//   * Nk_i = max(1, int(|C_i| / dl)),  n_i in [-Nk_i, Nk_i].
//   * mask 0 < k^2 <= (2*pi/dl)^2; keep one hemisphere (first non-zero n
//     component positive); factor = 2 for n != 0.
//   * kfac = exp(-sigma^2*k^2/2) / k^2.
//   * E = (norm_factor/V) * sum_k factor*kfac*(S_re^2 + S_im^2)
//         - norm_factor * sum_i q_i^2 / (sigma*(2*pi)^1.5)   [remove_self]
//     with S_re(k) = sum_i q_i cos(k.r_i), S_im(k) = sum_i q_i sin(k.r_i).
//   * F_i = -dE/dr_i
//         = (2*norm_factor/V) * sum_k factor*kfac*q_i
//           * (S_re*sin(k.r_i) - S_im*cos(k.r_i)) * k.
//     norm_factor = 90.4756 = 2*pi*14.3996 (eV*A/e^2).
//   * dE/dq_i = (2*norm_factor/V) * sum_k factor*kfac
//               * (S_re*cos(k.r_i) + S_im*sin(k.r_i))
//               - 2*norm_factor*q_i / (sigma*(2*pi)^1.5)   [remove_self]
//     The charge gradient is exposed by
//     energy_forces_with_charge_gradient(); energy_forces() keeps its
//     original signature and does not compute it.
//   * The k=0 term is excluded exactly as in the reference. For a non-neutral
//     system the reference omits the k=0/surface correction as well; callers
//     should pass a neutral charge set (or accept that convention).
//   * One charge channel per call. The reference sums independent charge
//     channels; for multiple channels call once per channel and add the
//     energies/forces.
//
// Engine integration
//   * Positions are accepted either as a flat interleaved 3N view
//     (Kokkos::View<const Precision*>, x fastest - the MACEKokkos `xyz`
//     convention) or as a rank-2 (N,3) view (Kokkos::View<const Precision*[3]>).
//     Forces accept the matching flat 3N view or (N,3) view.
//     Component access is stride-agnostic (view(i,axis) or view(3*i+axis)),
//     so LayoutLeft/LayoutRight rank-2 views both work.
//   * Cell may be passed as `const Precision cell[3][3]` (row-major rows) or
//     as `std::span<const Precision>` of length 9 (the MACEKokkos cell type).
//   * The function OVERWRITES all force components of the requested nodes.
//
// Kokkos / performance
//   * Requires an initialized Kokkos session. Uses Kokkos::DefaultExecutionSpace
//     (an explicit execution-space instance overload is provided).
//   * The k-list is built on the host, once per distinct cell; it is cached in
//     the (mutable) instance and only rebuilt when the cell changes (exact
//     floating-point comparison) or set_params() is called. This keeps the
//     per-step cost free of host round trips except for the final scalar
//     energy returned by parallel_reduce.
//   * Structure factors currently use Kokkos::atomic_add into one accumulator
//     per k (O(N*K) atomics). A team-level reduction is the obvious follow-up
//     if profiling shows this dominates. The force kernel maps one thread per
//     node and loops over all k; this is efficient for the N >> 1 regime of
//     MACE evaluation but gives little parallelism for very small N (e.g. the
//     4-atom self-test).
//   * Atomic accumulation makes the structure factors (and therefore the
//     energy/forces) non-bitwise-reproducible at the roundoff level across
//     runs. Results are deterministic up to floating-point summation order.
//   * Not thread-safe: concurrent calls on the same instance race on the
//     cached k-list/scratch views. Use one instance per stream/thread.
//
// Self-test (analytic forces and charge gradient against finite differences
// of the host reference):
//     compile with -DSYMMETRIX_LES_EWALD_SELF_TEST and call
//         LesEwaldKokkos::run_self_test();
//     inside an initialized Kokkos session (returns 0 on success).
// -----------------------------------------------------------------------------

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace les_ewald_detail {

inline constexpr double kNormFactor = 90.4756;  // 2*pi*14.3996 eV*A/e^2
inline constexpr double kTwoPi = 6.283185307179586476925286766559;

// Compile-time sentinel passed to the shared kernel implementation when no
// charge gradient is requested. Kept at namespace scope because CUDA extended
// lambdas reject private nested types in their template arguments.
struct NoChargeGradientView {};

template <typename Precision, class View>
KOKKOS_INLINE_FUNCTION
Precision position_component(const View& positions, const int node, const int axis) {
    if constexpr (static_cast<int>(View::rank) == 2) {
        return static_cast<Precision>(positions(node, axis));
    } else {
        return static_cast<Precision>(positions(3*node + axis));
    }
}

template <typename Precision, class View>
KOKKOS_INLINE_FUNCTION
void set_force_component(const View& forces, const int node, const int axis,
                         const Precision value) {
    if constexpr (static_cast<int>(View::rank) == 2) {
        forces(node, axis) = value;
    } else {
        forces(3*node + axis) = value;
    }
}

// The charge-gradient pieces live in device helper functions rather than in
// `if constexpr` blocks inside the node kernel: nvcc's extended lambdas
// cannot first-capture a variable inside a constexpr-if context.
template <bool WriteChargeGradient, typename Precision, class ScalarView>
KOKKOS_INLINE_FUNCTION
void accumulate_charge_gradient(const ScalarView& kweight,
                                const ScalarView& s_re,
                                const ScalarView& s_im,
                                const int kk,
                                const Precision cos_kr,
                                const Precision sin_kr,
                                Precision& grad) {
    if constexpr (WriteChargeGradient) {
        grad += kweight(kk)*(s_re(kk)*cos_kr + s_im(kk)*sin_kr);
    } else {
        (void)kweight;
        (void)s_re;
        (void)s_im;
        (void)kk;
        (void)cos_kr;
        (void)sin_kr;
        (void)grad;
    }
}

template <bool WriteChargeGradient, typename Precision, class ChargeGradView>
KOKKOS_INLINE_FUNCTION
void store_charge_gradient(const ChargeGradView& charge_grad,
                           const int node,
                           Precision value,
                           const Precision force_scale,
                           const bool remove_self,
                           const Precision self_scale,
                           const Precision q) {
    if constexpr (WriteChargeGradient) {
        value *= force_scale;
        if (remove_self)
            value -= self_scale*q;
        charge_grad(node) = value;
    } else {
        (void)charge_grad;
        (void)node;
        (void)value;
        (void)force_scale;
        (void)remove_self;
        (void)self_scale;
        (void)q;
    }
}

// Shared kernel body behind LesEwaldKokkosT::energy_forces() and
// energy_forces_with_charge_gradient(). Lives outside the class because CUDA
// extended lambdas cannot be defined in private/protected member functions.
// The caller has already built the k-list and resized the structure-factor
// scratch views.
template <bool WriteChargeGradient, typename Precision,
          class PositionView, class ChargeView, class ForceView,
          class ChargeGradView, class KVectorView, class ScalarView>
Precision compute_energy_forces_impl(
    const Kokkos::DefaultExecutionSpace& execution_space,
    const int num_nodes,
    const PositionView& positions,
    const ChargeView& charges,
    const ForceView& forces,
    const ChargeGradView& charge_grad,
    const KVectorView& k_vectors,
    const ScalarView& k_weights,
    const ScalarView& s_re,
    const ScalarView& s_im,
    const int num_k,
    const double volume,
    const bool remove_self,
    const double self_coeff,
    const double norm_factor) {
    const auto kvec = k_vectors;
    const auto kweight = k_weights;
    const auto s_re_view = s_re;
    const auto s_im_view = s_im;
    const auto position_view = positions;
    const auto charge_view = charges;
    const auto force_view = forces;
    const auto charge_grad_view = charge_grad;
    const Precision self_scale =
        static_cast<Precision>(2.0*norm_factor*self_coeff);

    // ---- 1. Structure factors S(k) = sum_i q_i exp(i k.r_i) -------------
    if (num_k > 0) {
        Kokkos::deep_copy(execution_space, s_re_view, Precision(0));
        Kokkos::deep_copy(execution_space, s_im_view, Precision(0));
        Kokkos::parallel_for(
            "LesEwald::structure_factors",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                execution_space, {0, 0}, {num_k, num_nodes}),
            KOKKOS_LAMBDA(const int kk, const int node) {
                const Precision kr =
                    kvec(kk, 0)*position_component<Precision>(
                        position_view, node, 0)
                    + kvec(kk, 1)*position_component<Precision>(
                        position_view, node, 1)
                    + kvec(kk, 2)*position_component<Precision>(
                        position_view, node, 2);
                const Precision q = static_cast<Precision>(charge_view(node));
                Kokkos::atomic_add(&s_re_view(kk), q*Kokkos::cos(kr));
                Kokkos::atomic_add(&s_im_view(kk), q*Kokkos::sin(kr));
            });
    }

    // ---- 2. Reciprocal energy (norm_factor/V) sum_k w_k |S_k|^2 ---------
    Precision k_sum = Precision(0);
    if (num_k > 0) {
        Kokkos::parallel_reduce(
            "LesEwald::reciprocal_energy",
            Kokkos::RangePolicy<>(execution_space, 0, num_k),
            KOKKOS_LAMBDA(const int kk, Precision& acc) {
                const Precision sr = s_re_view(kk);
                const Precision si = s_im_view(kk);
                acc += kweight(kk)*(sr*sr + si*si);
            },
            k_sum);
    }
    Precision energy =
        static_cast<Precision>(norm_factor/volume)*k_sum;

    // ---- 3. Self-interaction subtraction --------------------------------
    if (remove_self) {
        Precision q_sq = Precision(0);
        Kokkos::parallel_reduce(
            "LesEwald::self_energy",
            Kokkos::RangePolicy<>(execution_space, 0, num_nodes),
            KOKKOS_LAMBDA(const int node, Precision& acc) {
                const Precision q = static_cast<Precision>(charge_view(node));
                acc += q*q;
            },
            q_sq);
        energy -= static_cast<Precision>(norm_factor*self_coeff)*q_sq;
    }

    // ---- 4. Analytic forces (and optional charge gradient) --------------
    // F_i = (2*norm_factor/V) sum_k w_k q_i
    //       * (S_re sin(k.r_i) - S_im cos(k.r_i)) * k
    // dE/dq_i = (2*norm_factor/V) sum_k w_k
    //       * (S_re cos(k.r_i) + S_im sin(k.r_i))
    //       - 2*norm_factor*self_coeff*q_i   [remove_self]
    const Precision force_scale =
        static_cast<Precision>(2.0*norm_factor/volume);
    Kokkos::parallel_for(
        "LesEwald::forces",
        Kokkos::RangePolicy<>(execution_space, 0, num_nodes),
        KOKKOS_LAMBDA(const int node) {
            const Precision q = static_cast<Precision>(charge_view(node));
            const Precision x = position_component<Precision>(
                position_view, node, 0);
            const Precision y = position_component<Precision>(
                position_view, node, 1);
            const Precision z = position_component<Precision>(
                position_view, node, 2);
            Precision fx = Precision(0);
            Precision fy = Precision(0);
            Precision fz = Precision(0);
            Precision charge_grad_value = Precision(0);
            for (int kk = 0; kk < num_k; ++kk) {
                const Precision kr =
                    kvec(kk, 0)*x + kvec(kk, 1)*y + kvec(kk, 2)*z;
                const Precision sin_kr = Kokkos::sin(kr);
                const Precision cos_kr = Kokkos::cos(kr);
                const Precision c = kweight(kk)*q
                    *(s_re_view(kk)*sin_kr - s_im_view(kk)*cos_kr);
                fx += c*kvec(kk, 0);
                fy += c*kvec(kk, 1);
                fz += c*kvec(kk, 2);
                accumulate_charge_gradient<WriteChargeGradient>(
                    kweight, s_re_view, s_im_view, kk, cos_kr, sin_kr,
                    charge_grad_value);
            }
            set_force_component<Precision>(
                force_view, node, 0, force_scale*fx);
            set_force_component<Precision>(
                force_view, node, 1, force_scale*fy);
            set_force_component<Precision>(
                force_view, node, 2, force_scale*fz);
            store_charge_gradient<WriteChargeGradient>(
                charge_grad_view, node, charge_grad_value, force_scale,
                remove_self, self_scale, q);
        });

    return energy;
}

}  // namespace les_ewald_detail

template <typename Precision>
class LesEwaldKokkosT {
public:
    using precision_type = Precision;
    using scalar_view = Kokkos::View<Precision*>;
    using k_vector_view = Kokkos::View<Precision*[3], Kokkos::LayoutRight>;

    struct HostResult {
        double energy = 0.0;
        std::vector<double> forces;  // 3N, empty unless requested
    };

    struct HostResultWithChargeGradient {
        double energy = 0.0;
        std::vector<double> forces;           // 3N, empty unless requested
        std::vector<double> charge_gradient;  // N, empty unless requested
    };

    LesEwaldKokkosT() = default;

    LesEwaldKokkosT(const double sigma, const double dl,
                    const bool remove_self) {
        set_params(sigma, dl, remove_self);
    }

    void set_params(const double sigma, const double dl, const bool remove_self) {
        if (!std::isfinite(sigma) || !(sigma > 0.0))
            throw std::invalid_argument("LesEwald: sigma must be finite and > 0.");
        if (!std::isfinite(dl) || !(dl > 0.0))
            throw std::invalid_argument("LesEwald: dl must be finite and > 0.");
        if (sigma != sigma_ || dl != dl_ || !k_cache_valid_)
            k_cache_valid_ = false;
        sigma_ = sigma;
        dl_ = dl;
        remove_self_ = remove_self;
        sigma_sq_half_ = 0.5*sigma*sigma;
        const double cutoff = les_ewald_detail::kTwoPi/dl;
        k_sq_max_ = cutoff*cutoff;
        self_coeff_ = 1.0/(sigma*std::pow(les_ewald_detail::kTwoPi, 1.5));
    }

    double sigma() const noexcept { return sigma_; }
    double dl() const noexcept { return dl_; }
    bool remove_self() const noexcept { return remove_self_; }

    // Coulomb constant used by the model (2*pi*14.3996 = 90.4756 for the
    // reference convention; MACELES checkpoints store their own
    // les_norm_factor, e.g. 90.0474, and the engine must honour it).
    void set_norm_factor(const double value) {
        if (!std::isfinite(value) || !(value > 0.0))
            throw std::invalid_argument(
                "LesEwald: norm_factor must be finite and > 0.");
        norm_factor_ = value;
    }
    double norm_factor() const noexcept { return norm_factor_; }

    // Number of reciprocal vectors in the cached k-list for the last cell.
    // Returns 0 before the first energy_forces() call.
    std::size_t kpoint_count() const noexcept {
        return k_cache_valid_
            ? static_cast<std::size_t>(k_vectors_.extent(0)) : 0u;
    }

    // ------------------------------------------------------------------
    // Primary entry point. `cell` is const Precision[3][3], row-major,
    // lattice vectors on the rows. Returns the total monopole energy (eV)
    // and overwrites the force components of the first `num_nodes` nodes.
    // ------------------------------------------------------------------
    template <class PositionView, class ChargeView, class ForceView>
    Precision energy_forces(const int num_nodes,
                            const PositionView& positions,
                            const Precision cell[3][3],
                            const ChargeView& charges,
                            const ForceView& forces) const {
        return energy_forces(Kokkos::DefaultExecutionSpace(), num_nodes,
                             positions, cell, charges, forces);
    }

    template <class PositionView, class ChargeView, class ForceView>
    Precision energy_forces(const Kokkos::DefaultExecutionSpace& execution_space,
                            const int num_nodes,
                            const PositionView& positions,
                            const Precision cell[3][3],
                            const ChargeView& charges,
                            const ForceView& forces) const {
        return energy_forces_impl<false>(
            execution_space, num_nodes, positions, cell, charges, forces,
            les_ewald_detail::NoChargeGradientView{});
    }

    // ------------------------------------------------------------------
    // Same as energy_forces(), and additionally fills
    //     charge_grad(i) = dE/dq_i
    //         = (2*norm_factor/V) sum_k w_k
    //               * (S_re cos(k.r_i) + S_im sin(k.r_i))
    //           - 2*norm_factor*self_coeff*q_i   [remove_self only]
    // `charge_grad` is a rank-1 view of length >= num_nodes. The k=0 term is
    // excluded, matching the energy/force conventions above.
    // ------------------------------------------------------------------
    template <class PositionView, class ChargeView, class ForceView,
              class ChargeGradView>
    Precision energy_forces_with_charge_gradient(
        const int num_nodes,
        const PositionView& positions,
        const Precision cell[3][3],
        const ChargeView& charges,
        const ForceView& forces,
        const ChargeGradView& charge_grad) const {
        return energy_forces_with_charge_gradient(
            Kokkos::DefaultExecutionSpace(), num_nodes, positions, cell,
            charges, forces, charge_grad);
    }

    template <class PositionView, class ChargeView, class ForceView,
              class ChargeGradView>
    Precision energy_forces_with_charge_gradient(
        const Kokkos::DefaultExecutionSpace& execution_space,
        const int num_nodes,
        const PositionView& positions,
        const Precision cell[3][3],
        const ChargeView& charges,
        const ForceView& forces,
        const ChargeGradView& charge_grad) const {
        return energy_forces_impl<true>(
            execution_space, num_nodes, positions, cell, charges, forces,
            charge_grad);
    }

    // Convenience overload for the engine's std::span<const double> cell.
    template <class PositionView, class ChargeView, class ForceView>
    Precision energy_forces(const int num_nodes,
                            const PositionView& positions,
                            const std::span<const Precision> cell_row_major,
                            const ChargeView& charges,
                            const ForceView& forces) const {
        return energy_forces(Kokkos::DefaultExecutionSpace(), num_nodes,
                             positions, cell_row_major, charges, forces);
    }

    template <class PositionView, class ChargeView, class ForceView>
    Precision energy_forces(const Kokkos::DefaultExecutionSpace& execution_space,
                            const int num_nodes,
                            const PositionView& positions,
                            const std::span<const Precision> cell_row_major,
                            const ChargeView& charges,
                            const ForceView& forces) const {
        if (cell_row_major.size() != 9)
            throw std::invalid_argument(
                "LesEwald: cell span must contain 9 row-major entries.");
        Precision cell[3][3];
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                cell[row][column] = cell_row_major[3*row + column];
        return energy_forces(execution_space, num_nodes, positions, cell,
                             charges, forces);
    }

    // ------------------------------------------------------------------
    // Host reference (double precision, no Kokkos). Useful for regression
    // tests and for the finite-difference self-test below. positions is
    // flat 3N (x fastest), cell_row_major is 9 entries with lattice vectors
    // on the rows, charges is N.
    // ------------------------------------------------------------------
    static HostResult host_reference_energy_forces(
        const std::size_t num_nodes,
        const double* positions,
        const double* cell_row_major,
        const double* charges,
        const double sigma,
        const double dl,
        const bool remove_self,
        const bool compute_forces = true) {
        const HostResultWithChargeGradient full =
            host_reference_energy_forces_and_charge_gradient(
                num_nodes, positions, cell_row_major, charges, sigma, dl,
                remove_self, compute_forces, false);
        HostResult result;
        result.energy = full.energy;
        result.forces = full.forces;
        return result;
    }

    // Same as host_reference_energy_forces(), plus the analytic charge
    // gradient dE/dq_i:
    //     (2*norm_factor/V) sum_k w_k (S_re cos(k.r_i) + S_im sin(k.r_i))
    //     - 2*norm_factor*self_coeff*q_i   [remove_self only]
    static HostResultWithChargeGradient
    host_reference_energy_forces_and_charge_gradient(
        const std::size_t num_nodes,
        const double* positions,
        const double* cell_row_major,
        const double* charges,
        const double sigma,
        const double dl,
        const bool remove_self,
        const bool compute_forces = true,
        const bool compute_charge_gradient = true,
        const double norm_factor = les_ewald_detail::kNormFactor) {
        if (cell_row_major == nullptr)
            throw std::invalid_argument("LesEwald reference: null cell.");
        if (num_nodes != 0 && (positions == nullptr || charges == nullptr))
            throw std::invalid_argument("LesEwald reference: null positions/charges.");

        const HostKList klist = build_k_list_host(cell_row_major, sigma, dl);
        const std::size_t num_k = klist.kx.size();

        std::vector<double> s_re(num_k, 0.0);
        std::vector<double> s_im(num_k, 0.0);
        std::vector<double> cos_kr;
        std::vector<double> sin_kr;
        const bool need_per_node_trig = compute_forces || compute_charge_gradient;
        if (need_per_node_trig) {
            cos_kr.resize(num_nodes*num_k);
            sin_kr.resize(num_nodes*num_k);
        }
        for (std::size_t i = 0; i < num_nodes; ++i) {
            const double x = positions[3*i];
            const double y = positions[3*i + 1];
            const double z = positions[3*i + 2];
            const double q = charges[i];
            for (std::size_t kk = 0; kk < num_k; ++kk) {
                const double kr = klist.kx[kk]*x + klist.ky[kk]*y + klist.kz[kk]*z;
                const double c = std::cos(kr);
                const double s = std::sin(kr);
                s_re[kk] += q*c;
                s_im[kk] += q*s;
                if (need_per_node_trig) {
                    cos_kr[i*num_k + kk] = c;
                    sin_kr[i*num_k + kk] = s;
                }
            }
        }

        HostResultWithChargeGradient result;
        double k_sum = 0.0;
        for (std::size_t kk = 0; kk < num_k; ++kk)
            k_sum += klist.weight[kk]*(s_re[kk]*s_re[kk] + s_im[kk]*s_im[kk]);
        result.energy = norm_factor/klist.volume*k_sum;
        if (remove_self) {
            double q_sq = 0.0;
            for (std::size_t i = 0; i < num_nodes; ++i)
                q_sq += charges[i]*charges[i];
            result.energy -= norm_factor*q_sq
                /(sigma*std::pow(les_ewald_detail::kTwoPi, 1.5));
        }

        const double force_scale =
            2.0*norm_factor/klist.volume;

        if (compute_forces) {
            result.forces.assign(3*num_nodes, 0.0);
            for (std::size_t i = 0; i < num_nodes; ++i) {
                const double q = charges[i];
                double fx = 0.0;
                double fy = 0.0;
                double fz = 0.0;
                for (std::size_t kk = 0; kk < num_k; ++kk) {
                    const double c = klist.weight[kk]*q
                        *(s_re[kk]*sin_kr[i*num_k + kk]
                          - s_im[kk]*cos_kr[i*num_k + kk]);
                    fx += c*klist.kx[kk];
                    fy += c*klist.ky[kk];
                    fz += c*klist.kz[kk];
                }
                result.forces[3*i] = force_scale*fx;
                result.forces[3*i + 1] = force_scale*fy;
                result.forces[3*i + 2] = force_scale*fz;
            }
        }

        if (compute_charge_gradient) {
            result.charge_gradient.assign(num_nodes, 0.0);
            const double self_scale =
                2.0*norm_factor
                /(sigma*std::pow(les_ewald_detail::kTwoPi, 1.5));
            for (std::size_t i = 0; i < num_nodes; ++i) {
                double grad = 0.0;
                for (std::size_t kk = 0; kk < num_k; ++kk) {
                    grad += klist.weight[kk]
                        *(s_re[kk]*cos_kr[i*num_k + kk]
                          + s_im[kk]*sin_kr[i*num_k + kk]);
                }
                grad *= force_scale;
                if (remove_self)
                    grad -= self_scale*charges[i];
                result.charge_gradient[i] = grad;
            }
        }
        return result;
    }

    static double host_reference_energy(
        const std::size_t num_nodes,
        const double* positions,
        const double* cell_row_major,
        const double* charges,
        const double sigma,
        const double dl,
        const bool remove_self) {
        return host_reference_energy_forces(
            num_nodes, positions, cell_row_major, charges, sigma, dl,
            remove_self, false).energy;
    }

#ifdef SYMMETRIX_LES_EWALD_SELF_TEST
    // Finite-difference check of the analytic forces and charge gradient on
    // a deterministic 4-atom triclinic cell. Needs an initialized Kokkos
    // session. Returns 0 on success, 1 on failure.
    static int run_self_test(const bool verbose = true) {
        constexpr int num_nodes = 4;
        const Precision cell[3][3] = {
            {Precision(5.6), Precision(0.0), Precision(0.0)},
            {Precision(1.1), Precision(5.9), Precision(0.0)},
            {Precision(-0.7), Precision(0.9), Precision(6.3)}};
        const double host_positions[3*num_nodes] = {
            0.4, 0.3, 0.2,
            2.1, 1.2, 0.7,
            3.5, 4.4, 2.3,
            -1.0, 2.8, 5.8};
        const double host_charges[num_nodes] = {0.7, -0.9, 0.4, -0.2};
        const double sigma = 1.1;
        const double dl = 1.8;

        LesEwaldKokkosT<Precision> ewald(sigma, dl, true);

        Kokkos::View<Precision*[3]> positions("LesEwald::test positions", num_nodes);
        Kokkos::View<Precision*> charges("LesEwald::test charges", num_nodes);
        Kokkos::View<Precision*[3]> forces("LesEwald::test forces", num_nodes);
        Kokkos::View<Precision*[3]> forces_with_charge_grad(
            "LesEwald::test forces with charge grad", num_nodes);
        Kokkos::View<Precision*> charge_grad("LesEwald::test charge grad", num_nodes);
        {
            auto hp = Kokkos::create_mirror_view(positions);
            auto hq = Kokkos::create_mirror_view(charges);
            for (int i = 0; i < num_nodes; ++i) {
                for (int axis = 0; axis < 3; ++axis)
                    hp(i, axis) = static_cast<Precision>(host_positions[3*i + axis]);
                hq(i) = static_cast<Precision>(host_charges[i]);
            }
            Kokkos::deep_copy(positions, hp);
            Kokkos::deep_copy(charges, hq);
        }

        // Existing API (must be unchanged) and the new gradient API on the
        // same inputs; their energy/force results should be identical.
        const double energy = static_cast<double>(
            ewald.energy_forces(num_nodes, positions, cell, charges, forces));
        const double energy_with_charge_grad = static_cast<double>(
            ewald.energy_forces_with_charge_gradient(
                num_nodes, positions, cell, charges, forces_with_charge_grad,
                charge_grad));
        const auto host_forces = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), forces);
        const auto host_forces_with_charge_grad =
            Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), forces_with_charge_grad);
        const auto host_charge_grad = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), charge_grad);

        double max_api_force_diff = 0.0;
        for (int i = 0; i < num_nodes; ++i)
            for (int axis = 0; axis < 3; ++axis)
                max_api_force_diff = std::max(
                    max_api_force_diff,
                    std::abs(static_cast<double>(host_forces(i, axis))
                             - static_cast<double>(
                                 host_forces_with_charge_grad(i, axis))));
        const double api_energy_diff =
            std::abs(energy - energy_with_charge_grad);

        double cell_row_major[9];
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                cell_row_major[3*row + column] = static_cast<double>(cell[row][column]);

        const HostResultWithChargeGradient reference =
            host_reference_energy_forces_and_charge_gradient(
                num_nodes, host_positions, cell_row_major, host_charges,
                sigma, dl, true);
        const double reference_energy = reference.energy;

        // ---- forces: analytic vs central difference in position ----------
        const double step = 1e-5;
        double max_force_error = 0.0;
        for (int i = 0; i < num_nodes; ++i) {
            for (int axis = 0; axis < 3; ++axis) {
                std::vector<double> plus(host_positions, host_positions + 3*num_nodes);
                std::vector<double> minus(plus);
                plus[3*i + axis] += step;
                minus[3*i + axis] -= step;
                const double e_plus = host_reference_energy(
                    num_nodes, plus.data(), cell_row_major, host_charges,
                    sigma, dl, true);
                const double e_minus = host_reference_energy(
                    num_nodes, minus.data(), cell_row_major, host_charges,
                    sigma, dl, true);
                const double finite_difference = -(e_plus - e_minus)/(2.0*step);
                const double error = std::abs(
                    static_cast<double>(host_forces_with_charge_grad(i, axis))
                    - finite_difference);
                if (error > max_force_error)
                    max_force_error = error;
            }
        }

        // ---- charge gradient: analytic vs central difference in q --------
        const double charge_step = 1e-4;
        double max_charge_grad_abs_error = 0.0;
        double max_charge_grad_rel_error = 0.0;
        double max_charge_grad_host_error = 0.0;
        double max_charge_grad_magnitude = 0.0;
        for (int i = 0; i < num_nodes; ++i) {
            std::vector<double> plus(host_charges, host_charges + num_nodes);
            std::vector<double> minus(plus);
            plus[i] += charge_step;
            minus[i] -= charge_step;
            const double e_plus = host_reference_energy(
                num_nodes, host_positions, cell_row_major, plus.data(),
                sigma, dl, true);
            const double e_minus = host_reference_energy(
                num_nodes, host_positions, cell_row_major, minus.data(),
                sigma, dl, true);
            const double finite_difference =
                (e_plus - e_minus)/(2.0*charge_step);
            const double analytic =
                static_cast<double>(host_charge_grad(i));
            const double abs_error = std::abs(analytic - finite_difference);
            const double denominator = std::max(
                std::abs(analytic), std::abs(finite_difference));
            const double rel_error =
                denominator > 0.0 ? abs_error/denominator : abs_error;
            if (abs_error > max_charge_grad_abs_error)
                max_charge_grad_abs_error = abs_error;
            if (rel_error > max_charge_grad_rel_error)
                max_charge_grad_rel_error = rel_error;
            max_charge_grad_host_error = std::max(
                max_charge_grad_host_error,
                std::abs(analytic - reference.charge_gradient[i]));
            max_charge_grad_magnitude = std::max(
                max_charge_grad_magnitude, std::abs(analytic));
        }

        const double energy_error = std::abs(energy - reference_energy);
        const double tolerance = std::is_same_v<Precision, float> ? 1e-3 : 1e-6;
        const bool ok =
            max_force_error <= tolerance
            && energy_error <= tolerance
            && max_charge_grad_abs_error <= tolerance
            && max_charge_grad_rel_error <= tolerance
            && max_charge_grad_host_error <= tolerance
            && max_api_force_diff <= tolerance
            && api_energy_diff <= tolerance;
        if (verbose) {
            std::printf(
                "LesEwald self-test: energy=%.12f reference=%.12f "
                "max|F_analytic-F_fd|=%.3e "
                "max|dE/dq_analytic-dE/dq_fd|=%.3e "
                "max rel=%.3e max|dE/dq_kokkos-dE/dq_ref|=%.3e "
                "max|dE/dq|=%.3e max|F_api-F_new|=%.3e tolerance=%.1e "
                "-> %s\n",
                energy, reference_energy, max_force_error,
                max_charge_grad_abs_error, max_charge_grad_rel_error,
                max_charge_grad_host_error, max_charge_grad_magnitude,
                max_api_force_diff, tolerance, ok ? "PASS" : "FAIL");
        }
        return ok ? 0 : 1;
    }
#endif  // SYMMETRIX_LES_EWALD_SELF_TEST

private:
    template <bool WriteChargeGradient, class PositionView, class ChargeView,
              class ForceView, class ChargeGradView>
    Precision energy_forces_impl(
        const Kokkos::DefaultExecutionSpace& execution_space,
        const int num_nodes,
        const PositionView& positions,
        const Precision cell[3][3],
        const ChargeView& charges,
        const ForceView& forces,
        const ChargeGradView& charge_grad) const {
        static_assert(static_cast<int>(PositionView::rank) == 1
                          || static_cast<int>(PositionView::rank) == 2,
                      "LesEwald: positions must be a flat 3N view or an (N,3) view.");
        static_assert(static_cast<int>(ChargeView::rank) == 1,
                      "LesEwald: charges must be a rank-1 (N) view.");
        static_assert(static_cast<int>(ForceView::rank) == 1
                          || static_cast<int>(ForceView::rank) == 2,
                      "LesEwald: forces must be a flat 3N view or an (N,3) view.");
        static_assert(std::is_arithmetic_v<typename PositionView::non_const_value_type>
                          && std::is_arithmetic_v<typename ChargeView::non_const_value_type>
                          && std::is_arithmetic_v<typename ForceView::non_const_value_type>,
                      "LesEwald: views must hold arithmetic (floating-point) values.");
        if constexpr (WriteChargeGradient) {
            static_assert(static_cast<int>(ChargeGradView::rank) == 1,
                          "LesEwald: charge_grad must be a rank-1 (N) view.");
            static_assert(
                std::is_arithmetic_v<typename ChargeGradView::non_const_value_type>,
                "LesEwald: charge_grad must hold arithmetic (floating-point) values.");
        }

        if (num_nodes < 0)
            throw std::invalid_argument("LesEwald: num_nodes must be non-negative.");
        const std::size_t num = static_cast<std::size_t>(num_nodes);
        if (node_count(positions) < num)
            throw std::invalid_argument("LesEwald: positions extent is smaller than num_nodes.");
        if (charges.extent(0) < num)
            throw std::invalid_argument("LesEwald: charges extent is smaller than num_nodes.");
        if (node_count(forces) < num)
            throw std::invalid_argument("LesEwald: forces extent is smaller than num_nodes.");
        if constexpr (WriteChargeGradient) {
            if (charge_grad.extent(0) < num)
                throw std::invalid_argument(
                    "LesEwald: charge_grad extent is smaller than num_nodes.");
        }
        if (num == 0)
            return Precision(0);

        ensure_k_list(execution_space, cell);
        const int num_k = static_cast<int>(k_vectors_.extent(0));

        if (s_re_.extent(0) != static_cast<std::size_t>(num_k)) {
            s_re_ = scalar_view("LesEwald::S_re", num_k);
            s_im_ = scalar_view("LesEwald::S_im", num_k);
        }

        return les_ewald_detail::compute_energy_forces_impl<
            WriteChargeGradient, Precision>(
            execution_space, num_nodes, positions, charges, forces,
            charge_grad, k_vectors_, k_weights_, s_re_, s_im_, num_k,
            cached_volume_, remove_self_, self_coeff_, norm_factor_);
    }

    struct HostKList {
        double volume = 1.0;
        std::vector<double> kx, ky, kz, weight;
    };

    static HostKList build_k_list_host(const double cell_row_major[9],
                                       const double sigma, const double dl) {
        if (!std::isfinite(sigma) || !(sigma > 0.0))
            throw std::invalid_argument("LesEwald: sigma must be finite and > 0.");
        if (!std::isfinite(dl) || !(dl > 0.0))
            throw std::invalid_argument("LesEwald: dl must be finite and > 0.");
        for (int i = 0; i < 9; ++i)
            if (!std::isfinite(cell_row_major[i]))
                throw std::invalid_argument("LesEwald: cell contains a non-finite value.");

        const double a = cell_row_major[0];
        const double b = cell_row_major[1];
        const double c = cell_row_major[2];
        const double d = cell_row_major[3];
        const double e = cell_row_major[4];
        const double f = cell_row_major[5];
        const double g = cell_row_major[6];
        const double h = cell_row_major[7];
        const double i = cell_row_major[8];
        const double det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
        if (!std::isfinite(det) || det == 0.0)
            throw std::invalid_argument("LesEwald: cell matrix is not invertible.");

        HostKList result;
        result.volume = std::abs(det);
        const double inv = 1.0/det;
        // Row-major inverse of the row-major cell matrix.
        const double m00 = (e*i - f*h)*inv;
        const double m01 = (c*h - b*i)*inv;
        const double m02 = (b*f - c*e)*inv;
        const double m10 = (f*g - d*i)*inv;
        const double m11 = (a*i - c*g)*inv;
        const double m12 = (c*d - a*f)*inv;
        const double m20 = (d*h - e*g)*inv;
        const double m21 = (b*g - a*h)*inv;
        const double m22 = (a*e - b*d)*inv;
        // G = 2*pi*inv(C)^T, i.e. G[i][j] = 2*pi*inv_cell[j][i].
        const double two_pi = les_ewald_detail::kTwoPi;
        const double G[3][3] = {
            {two_pi*m00, two_pi*m10, two_pi*m20},
            {two_pi*m01, two_pi*m11, two_pi*m21},
            {two_pi*m02, two_pi*m12, two_pi*m22}};

        int Nk[3];
        for (int axis = 0; axis < 3; ++axis) {
            const double row_norm = std::sqrt(
                cell_row_major[3*axis]*cell_row_major[3*axis]
                + cell_row_major[3*axis + 1]*cell_row_major[3*axis + 1]
                + cell_row_major[3*axis + 2]*cell_row_major[3*axis + 2]);
            const double nk_real = row_norm/dl;
            if (!std::isfinite(nk_real) || nk_real > 1.0e9)
                throw std::runtime_error(
                    "LesEwald: cell/dl too large; increase dl.");
            Nk[axis] = static_cast<int>(nk_real);
            if (Nk[axis] < 1) Nk[axis] = 1;
        }
        const double grid_size =
            static_cast<double>(2*Nk[0] + 1)
            *static_cast<double>(2*Nk[1] + 1)
            *static_cast<double>(2*Nk[2] + 1);
        if (grid_size > 2.0e8)
            throw std::runtime_error(
                "LesEwald: k-grid too large for dl; increase dl.");

        const double k_sq_max = (two_pi/dl)*(two_pi/dl);
        const double sigma_sq_half = 0.5*sigma*sigma;
        const std::size_t reserve_hint =
            static_cast<std::size_t>(grid_size*0.25) + 1;
        result.kx.reserve(reserve_hint);
        result.ky.reserve(reserve_hint);
        result.kz.reserve(reserve_hint);
        result.weight.reserve(reserve_hint);

        for (int n1 = -Nk[0]; n1 <= Nk[0]; ++n1) {
            for (int n2 = -Nk[1]; n2 <= Nk[1]; ++n2) {
                for (int n3 = -Nk[2]; n3 <= Nk[2]; ++n3) {
                    const double kx = n1*G[0][0] + n2*G[1][0] + n3*G[2][0];
                    const double ky = n1*G[0][1] + n2*G[1][1] + n3*G[2][1];
                    const double kz = n1*G[0][2] + n2*G[1][2] + n3*G[2][2];
                    const double k_sq = kx*kx + ky*ky + kz*kz;
                    if (!(k_sq > 0.0) || k_sq > k_sq_max)
                        continue;
                    // One hemisphere: first non-zero component of n positive.
                    const int first_non_zero = n1 != 0 ? n1 : (n2 != 0 ? n2 : n3);
                    if (first_non_zero <= 0)
                        continue;
                    // n == 0 is already excluded by k_sq > 0; keep the
                    // reference's factor convention for clarity.
                    const double factor =
                        (n1 == 0 && n2 == 0 && n3 == 0) ? 1.0 : 2.0;
                    result.kx.push_back(kx);
                    result.ky.push_back(ky);
                    result.kz.push_back(kz);
                    result.weight.push_back(
                        factor*std::exp(-sigma_sq_half*k_sq)/k_sq);
                }
            }
        }
        return result;
    }

    template <class View>
    static std::size_t node_count(const View& view) {
        if constexpr (static_cast<int>(View::rank) == 2)
            return view.extent(0);
        else
            return view.extent(0)/3;
    }

    void ensure_k_list(const Kokkos::DefaultExecutionSpace& execution_space,
                       const Precision cell[3][3]) const {
        double flat[9];
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                flat[3*row + column] = static_cast<double>(cell[row][column]);
        if (k_cache_valid_) {
            bool same = true;
            for (int entry = 0; entry < 9; ++entry) {
                if (flat[entry] != cached_cell_[entry]) {
                    same = false;
                    break;
                }
            }
            if (same)
                return;
        }

        const HostKList klist = build_k_list_host(flat, sigma_, dl_);
        const std::size_t num_k = klist.kx.size();
        k_vectors_ = k_vector_view("LesEwald::k_vectors", num_k);
        k_weights_ = scalar_view("LesEwald::k_weights", num_k);
        auto host_k = Kokkos::create_mirror_view(k_vectors_);
        auto host_w = Kokkos::create_mirror_view(k_weights_);
        for (std::size_t kk = 0; kk < num_k; ++kk) {
            host_k(kk, 0) = static_cast<Precision>(klist.kx[kk]);
            host_k(kk, 1) = static_cast<Precision>(klist.ky[kk]);
            host_k(kk, 2) = static_cast<Precision>(klist.kz[kk]);
            host_w(kk) = static_cast<Precision>(klist.weight[kk]);
        }
        Kokkos::deep_copy(execution_space, k_vectors_, host_k);
        Kokkos::deep_copy(execution_space, k_weights_, host_w);
        for (int entry = 0; entry < 9; ++entry)
            cached_cell_[entry] = flat[entry];
        cached_volume_ = klist.volume;
        k_cache_valid_ = true;
    }

    double sigma_ = 1.0;
    double dl_ = 2.0;
    bool remove_self_ = true;
    double norm_factor_ = les_ewald_detail::kNormFactor;
    double sigma_sq_half_ = 0.5;
    double k_sq_max_ =
        les_ewald_detail::kTwoPi*les_ewald_detail::kTwoPi/4.0;
    double self_coeff_ = 1.0/std::pow(les_ewald_detail::kTwoPi, 1.5);

    // Cache of the reciprocal-space geometry. Mutable because energy_forces()
    // is const; rebuild happens only when the cell changes or set_params() is
    // called. Single-threaded use per instance (see header comment).
    mutable bool k_cache_valid_ = false;
    mutable std::array<double, 9> cached_cell_{};
    mutable double cached_volume_ = 1.0;
    mutable k_vector_view k_vectors_;
    mutable scalar_view k_weights_;
    mutable scalar_view s_re_;
    mutable scalar_view s_im_;
};

using LesEwaldKokkos = LesEwaldKokkosT<double>;
using LesEwaldFloatKokkos = LesEwaldKokkosT<float>;
