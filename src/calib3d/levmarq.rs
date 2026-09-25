/*
 *  levmarq.rs
 *  purecv
 *
 *  This file is part of purecv - WebARKit.
 *
 *  purecv is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  purecv is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with purecv.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  As a special exception, the copyright holders of this library give you
 *  permission to link this library with independent modules to produce an
 *  executable, regardless of the license terms of these independent modules, and to
 *  copy and distribute the resulting executable under terms of your choice,
 *  provided that you also meet, for each linked independent module, the terms and
 *  conditions of the license of that module. An independent module is a module
 *  which is neither derived from nor based on this library. If you modify this
 *  library, you may extend this exception to your version of the library, but you
 *  are not obligated to do so. If you do not wish to do so, delete this exception
 *  statement from your version.
 *
 *  Copyright 2026 WebARKit.
 *
 *  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
 *
 */

//! Levenberg-Marquardt solver — `LevMarq`.
//!
//! Minimises a sum-of-squares objective starting from an initial parameter
//! vector by damping the Gauss-Newton normal equations, ported from OpenCV's
//! `cv::LevMarq` (dense / linear-variable subset).
//!
//! Each probe solves `(JᵀJ + λ·D)·δ = −Jᵀb`, where `D = diag(JᵀJ)` is first
//! scaled by `λ` and clamped to `[1e-6, 1e32]`. Rejected probes inflate `λ`,
//! accepted ones deflate it (optionally scaled by the step quality, as in
//! OpenCV). The solver stops on the first of: step norm, relative energy change
//! or gradient norm below their tolerances, energy below
//! [`Settings::small_energy_tolerance`], `λ` reaching `1e32`, or
//! [`Settings::max_iterations`] probes. Only the first four count as
//! [`Report::found`] — exhausting iterations or `λ` does not.
//!
//! # Divergences from OpenCV
//!
//! | OpenCV | purecv |
//! |--------|--------|
//! | Dense *and* sparse matrices, linear/SO(3)/SE(3) variables | Dense matrices, linear variables only |
//! | "Long" (full Jacobian) and "normal" (`JᵀJ`, `Jᵀr`) callbacks | Normal callback only |
//! | Fixed-variable mask, Jacobi column scaling, geodesic acceleration | Not ported |
//! | `Settings` fields set through fluent setters | Plain `pub` fields, same defaults |
//!
//! # References
//!
//! Levenberg, K. (1944). *A method for the solution of certain non-linear
//! problems in least squares*. Marquardt, D. (1963). *An algorithm for least
//! squares estimation of nonlinear parameters*.

use alloc::vec;
use alloc::vec::Vec;
#[allow(unused_imports)]
use num_traits::Float;

use super::linalg::solve_dense;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Solver settings, mirroring `cv::LevMarq::Settings` defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Maximum number of probes (OpenCV default: 500).
    pub max_iterations: usize,
    /// Initial damping factor `λ` (OpenCV default: 1e-4).
    pub initial_lambda: f64,
    /// Factor `λ` is multiplied by when a probe is rejected (default: 2.0).
    pub lambda_up_factor: f64,
    /// Factor `λ` is divided by when a probe is accepted (default: 3.0).
    pub lambda_down_factor: f64,
    /// Double [`Settings::lambda_up_factor`] after each rejection (default: true).
    pub up_double: bool,
    /// Scale the `λ` decrease by the step quality (default: true).
    pub use_step_quality: bool,
    /// Clamp `λ·diag(JᵀJ)` to `[1e-6, 1e32]` (default: true).
    pub clamp_diagonal: bool,
    /// Measure the step with the ℓ∞ instead of the ℓ2 norm (default: false).
    pub step_norm_inf: bool,
    /// Stop when `Δenergy / energy` drops below the tolerance (default: true).
    pub check_rel_energy_change: bool,
    /// Stop when `‖Jᵀr‖∞` drops below the tolerance (default: true).
    pub check_min_gradient: bool,
    /// Stop when the step norm drops below the tolerance (default: true).
    pub check_step_norm: bool,
    /// Step-norm tolerance (OpenCV default: 1e-6).
    pub step_norm_tolerance: f64,
    /// Relative energy-change tolerance (OpenCV default: 1e-6).
    pub rel_energy_delta_tolerance: f64,
    /// Gradient-norm tolerance (OpenCV default: 1e-6).
    pub min_gradient_tolerance: f64,
    /// Energy tolerance; `0` disables it, as in OpenCV.
    pub small_energy_tolerance: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            initial_lambda: 1e-4,
            lambda_up_factor: 2.0,
            lambda_down_factor: 3.0,
            up_double: true,
            use_step_quality: true,
            clamp_diagonal: true,
            step_norm_inf: false,
            check_rel_energy_change: true,
            check_min_gradient: true,
            check_step_norm: true,
            step_norm_tolerance: 1e-6,
            rel_energy_delta_tolerance: 1e-6,
            min_gradient_tolerance: 1e-6,
            small_energy_tolerance: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Outcome of [`LevMarq::optimize`], mirroring `cv::LevMarq::Report`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Report {
    /// `true` when a convergence criterion was met. Exhausting the iteration
    /// budget or inflating `λ` past its limit reports `false`, not success.
    pub found: bool,
    /// Number of probes performed.
    pub iters: usize,
    /// Energy (sum of squared residuals) at the returned parameters.
    pub energy: f64,
}

// ---------------------------------------------------------------------------
// Callback
// ---------------------------------------------------------------------------

/// Objective function in the "normal equations" form.
///
/// Implemented by the caller because the solver is agnostic to how residuals
/// are produced. Both methods must be consistent: [`Callback::energy`] is the
/// same sum of squared residuals that [`Callback::compute`] reports alongside
/// `JᵀJ` / `Jᵀr`, and must skip the same degenerate terms.
pub trait Callback {
    /// Energy (sum of squared residuals) at `param`, without the Jacobian.
    /// `None` when the objective cannot be evaluated.
    fn energy(&mut self, param: &[f64]) -> Option<f64>;

    /// Energy plus the normal-equation matrix `jtj` (`nvars`×`nvars`,
    /// row-major) and the gradient `jtb` (`nvars`), both overwritten in full.
    /// `None` when the objective cannot be evaluated.
    fn compute(&mut self, param: &[f64], jtj: &mut [f64], jtb: &mut [f64]) -> Option<f64>;
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Dense, linear-variable Levenberg-Marquardt solver.
///
/// The solver owns its work buffers, so repeated [`LevMarq::optimize`] calls
/// (e.g. one per RANSAC hypothesis) allocate nothing after construction.
pub struct LevMarq {
    settings: Settings,
    nvars: usize,
    jtj: Vec<f64>,
    jtb: Vec<f64>,
    diag: Vec<f64>,
    lm_diag: Vec<f64>,
    step: Vec<f64>,
    probe: Vec<f64>,
    scratch: Vec<f64>,
}

impl LevMarq {
    /// Creates a solver for `nvars` parameters.
    pub fn new(nvars: usize, settings: Settings) -> Self {
        Self {
            settings,
            nvars,
            jtj: vec![0.0; nvars * nvars],
            jtb: vec![0.0; nvars],
            diag: vec![0.0; nvars],
            lm_diag: vec![0.0; nvars],
            step: vec![0.0; nvars],
            probe: vec![0.0; nvars],
            scratch: vec![0.0; nvars * nvars],
        }
    }

    /// Number of parameters this solver was created for.
    pub fn nvars(&self) -> usize {
        self.nvars
    }

    /// Settings this solver runs with.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Minimises `callback` starting from `param`, which is updated in place.
    ///
    /// `param` must have [`LevMarq::nvars`] elements; a length mismatch returns
    /// a not-found [`Report`] without touching it. On `Report::found == false`
    /// the parameters are left at the best point seen so far, which may be the
    /// initial guess.
    pub fn optimize<C: Callback>(&mut self, param: &mut [f64], callback: &mut C) -> Report {
        let n = self.nvars;
        if param.len() != n {
            return Report {
                found: false,
                iters: 0,
                energy: 0.0,
            };
        }

        let s = self.settings;
        // Diagonal clamp and λ ceiling, as in OpenCV.
        const MIN_DIAG: f64 = 1e-6;
        const MAX_DIAG: f64 = 1e32;
        const MAX_LAMBDA: f64 = 1e32;

        let mut energy = match callback.compute(param, &mut self.jtj, &mut self.jtb) {
            Some(e) if e.is_finite() && e >= 0.0 => e,
            _ => {
                return Report {
                    found: false,
                    iters: 0,
                    energy: 0.0,
                }
            }
        };
        let mut old_energy = energy;

        let mut lambda = s.initial_lambda;
        let mut lm_up_factor = s.lambda_up_factor;
        let mut iters = 0usize;
        let mut small_gradient = false;
        let mut small_step = false;
        let mut small_energy_delta = false;
        let mut small_energy = false;

        loop {
            // `self.jtj` / `self.jtb` hold the normal equations at `param`.
            let gradient_max = self.jtb.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for i in 0..n {
                self.diag[i] = self.jtj[i * n + i];
            }

            // Probe with increasing damping until a step lowers the energy.
            let mut step_norm = 0.0;
            loop {
                let mut cost_change = -1.0;

                for i in 0..n {
                    let mut d = self.diag[i] * lambda;
                    if s.clamp_diagonal {
                        d = d.clamp(MIN_DIAG, MAX_DIAG);
                    }
                    self.lm_diag[i] = d;
                }
                self.scratch.copy_from_slice(&self.jtj);
                for i in 0..n {
                    self.scratch[i * n + i] = self.diag[i] + self.lm_diag[i];
                    self.step[i] = -self.jtb[i];
                }
                // (JᵀJ + λ·D)·δ = −Jᵀb
                let solved = solve_dense(n, &mut self.scratch, &mut self.step);

                if solved {
                    step_norm = if s.step_norm_inf {
                        self.step.iter().fold(0.0f64, |m, v| m.max(v.abs()))
                    } else {
                        self.step.iter().map(|v| v * v).sum::<f64>().sqrt()
                    };
                    for (dst, (p, s)) in self
                        .probe
                        .iter_mut()
                        .zip(param.iter().zip(self.step.iter()))
                    {
                        *dst = p + s;
                    }
                    match callback.energy(&self.probe) {
                        Some(e) if e.is_finite() && e >= 0.0 => energy = e,
                        _ => {
                            return Report {
                                found: false,
                                iters,
                                energy: old_energy,
                            }
                        }
                    }
                    cost_change = old_energy - energy;
                }

                // A zero cost change counts as a failure when the relative
                // energy check is off, as in OpenCV.
                let rejected = !solved
                    || cost_change < 0.0
                    || (!s.check_rel_energy_change && cost_change.abs() < f64::EPSILON);

                if rejected {
                    lambda *= lm_up_factor;
                    if s.up_double {
                        lm_up_factor *= 2.0;
                    }
                } else {
                    let step_quality =
                        cost_change / jac_cost_change(&self.jtb, &self.step, &self.lm_diag);
                    lambda *= if s.use_step_quality {
                        (1.0 / s.lambda_down_factor).max(1.0 - (2.0 * step_quality - 1.0).powi(3))
                    } else {
                        1.0 / s.lambda_down_factor
                    };
                    lm_up_factor = s.lambda_up_factor;

                    // These flags stay set until the next accepted probe.
                    small_gradient = gradient_max < s.min_gradient_tolerance;
                    small_step = step_norm < s.step_norm_tolerance;
                    small_energy_delta = cost_change / energy < s.rel_energy_delta_tolerance;
                    small_energy = energy < s.small_energy_tolerance;

                    param.copy_from_slice(&self.probe);
                    old_energy = energy;
                }

                iters += 1;
                let done = iters >= s.max_iterations
                    || lambda >= MAX_LAMBDA
                    || (s.check_min_gradient && small_gradient)
                    || (s.check_step_norm && small_step)
                    || (s.check_rel_energy_change && small_energy_delta)
                    || small_energy;
                if done {
                    return Report {
                        found: small_gradient || small_step || small_energy_delta || small_energy,
                        iters,
                        energy: old_energy,
                    };
                }
                if !rejected {
                    break;
                }
            }

            // Normal equations at the accepted parameters for the next iteration.
            match callback.compute(param, &mut self.jtj, &mut self.jtb) {
                Some(e) if e.is_finite() && e >= 0.0 => {
                    energy = e;
                    old_energy = e;
                }
                _ => {
                    return Report {
                        found: false,
                        iters,
                        energy: old_energy,
                    }
                }
            }
        }
    }
}

/// Predicted energy reduction `−½·δᵀ(Jᵀb − λD·δ)`, OpenCV's
/// `calcJacCostChangeLm`, used to score the step quality.
fn jac_cost_change(jtb: &[f64], step: &[f64], lm_diag: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..jtb.len() {
        sum += step[i] * (jtb[i] - lm_diag[i] * step[i]);
    }
    -0.5 * sum
}
