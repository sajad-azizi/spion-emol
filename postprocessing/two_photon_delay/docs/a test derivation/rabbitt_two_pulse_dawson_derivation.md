# Upgrade of the one-pulse Eq. (7) to a two-pulse RABBITT Dawson kernel

## 1. Starting point

Let the initial bound state have energy

\[
E_i=-I_p,
\]

and let the final continuum energy be \(E\). For a two-photon process through intermediate states \(|k\rangle\), second-order perturbation theory gives

\[
a^{(2)}(E)=\sum_k d_{Ek}d_{ki}
\int_{-\infty}^{\infty}dt\,A_b(t)e^{i(E-E_k)t}
\int_{-\infty}^{t}dt'\,A_a(t')e^{i(E_k-E_i)t'} .
\]

Here pulse/component \(a\) acts first and pulse/component \(b\) acts second. The one-pulse result in Eq. (7) is recovered when both components come from the same Gaussian pulse and have opposite carrier signs.

## 2. Two-pulse Gaussian kernel

For the Gaussian envelopes only, define

\[
\mathcal K_{ba}(\beta,\alpha)
=
\int_{-\infty}^{\infty}dt\,
 e^{-(t-\tau_b)^2/T_b^2} e^{i\beta t}
\int_{-\infty}^{t}dt'\,
 e^{-(t'-\tau_a)^2/T_a^2} e^{i\alpha t'} .
\]

The two detunings are

\[
\alpha=E_k-E_i-s_a\omega_a,
\qquad
\beta=E-E_k-s_b\omega_b,
\]

where \(s=+1\) denotes absorption and \(s=-1\) denotes emission. The closed form is

\[
\boxed{
\mathcal K_{ba}(\beta,\alpha)=
\frac{\pi T_aT_b}{2}
\exp\!\left[-\frac{\alpha^2T_a^2}{4}-\frac{\beta^2T_b^2}{4}
+i\alpha\tau_a+i\beta\tau_b\right]
\left[1+\operatorname{erf} z\right]
}
\]

with

\[
\boxed{
 z=\frac{\tau_b-\tau_a+\frac{i}{2}(\beta T_b^2-\alpha T_a^2)}
 {\sqrt{T_a^2+T_b^2}}
}
\]

or, equivalently, as a complex Dawson kernel,

\[
\boxed{
1+\operatorname{erf}z
=
1+\frac{2i}{\sqrt\pi}e^{-z^2}F(-iz)
}
\]

where \(F\) is the Dawson function. When \(T_a=T_b=T\), \(\tau_a=\tau_b=0\), \(\alpha=\Delta_k^\eta+I_p\), and \(\beta=E-\Delta_k^\eta\), this becomes the one-pulse Eq. (7) kernel.

## 3. RABBITT specialization

Use an XUV harmonic comb centered at \(\tau_X=0\) and a delayed IR pulse centered at \(\tau_L=\tau\):

\[
A_X(t)=\sum_n\frac{A_n}{2}e^{-t^2/T_X^2}e^{-i(\Omega_nt+\phi_n)}+\text{c.c.},
\]

\[
A_L(t;\tau)=\frac{A_L}{2}e^{-(t-\tau)^2/T_L^2}
\left[e^{-i(\omega(t-\tau)+\phi_L)}+e^{+i(\omega(t-\tau)+\phi_L)}\right].
\]

For sideband \(2q\), the two resonant RABBITT paths are:

1. absorb harmonic \(\Omega_{2q-1}\), then absorb one IR photon;
2. absorb harmonic \(\Omega_{2q+1}\), then emit one IR photon.

Define

\[
\alpha_< = E_k+I_p-\Omega_{2q-1},
\qquad
\beta_< = E-E_k-\omega,
\]

\[
\alpha_> = E_k+I_p-\Omega_{2q+1},
\qquad
\beta_> = E-E_k+\omega.
\]

The energy-resolved RABBITT sideband amplitude is then

\[
\boxed{
\begin{aligned}
a_{2q}(E,\tau)=\frac{A_L}{4}\sum_k d_{Ek}d_{ki}
\Big[&A_{2q-1}e^{-i\phi_{2q-1}}e^{+i\omega\tau-i\phi_L}
\mathcal K_{LX}(\beta_<,\alpha_<;\tau,0,T_L,T_X)
\\
&+A_{2q+1}e^{-i\phi_{2q+1}}e^{-i\omega\tau+i\phi_L}
\mathcal K_{LX}(\beta_>,\alpha_>;\tau,0,T_L,T_X)
\Big] .
\end{aligned}
}
\]

If the harmonic amplitudes are absorbed into \(A_{2q\pm1}\), this is the direct RABBITT upgrade of Eq. (7). The corresponding spectrum is

\[
P_{2q}(E,\tau)=\rho_l(E)\,|a_{2q}(E,\tau)|^2,
\]

where near threshold one may use \(\rho_l(E)\propto E^{l+1/2}\).

## 4. RABBITT delay extracted from the kernel

Define the two complex path amplitudes

\[
M_< (E,\tau)=\sum_k d_{Ek}d_{ki}
\mathcal K_{LX}(\beta_<,\alpha_<;\tau,0,T_L,T_X),
\]

\[
M_> (E,\tau)=\sum_k d_{Ek}d_{ki}
\mathcal K_{LX}(\beta_>,\alpha_>;\tau,0,T_L,T_X).
\]

With the phase convention above, the sideband oscillation contains

\[
S_{2q}(\tau)=S_0+S_1
\cos\left[2\omega\tau+\phi_{2q+1}-\phi_{2q-1}-2\phi_L
+\arg\{M_<M_>^*\}\right].
\]

Therefore the finite-pulse two-photon RABBITT delay contribution is

\[
\boxed{
\tau_{2\hbar\omega}^{\rm kernel}(E,\tau)
=\frac{1}{2\omega}\arg\{M_< (E,\tau)M_>^*(E,\tau)\}
}
\]

up to the sign convention used for the Fourier components of the fields.

## 5. Numerical test

I tested the analytic kernel against direct quadrature of the original time-ordered integral. The numerical integral used the exact inner Gaussian integral and an adaptive outer quadrature. Five parameter sets were tested, including unequal XUV/IR pulse durations, finite delay, the zero-delay Eq. (7) limit, and off-resonant detunings.

Using physical RABBITT-like parameters, with times converted using \(\hbar=0.6582119514\,\mathrm{eV\,fs}\):

- \(T_{X,\mathrm{FWHM}}=0.35\,\mathrm{fs}\),
- \(T_{L,\mathrm{FWHM}}=5.0\,\mathrm{fs}\),
- \(\tau=0.45\,\mathrm{fs}\),
- representative detunings \(|\alpha|,|\beta|\lesssim 1\,\mathrm{eV}\).

The maximum observed relative errors were

\[
\max \frac{|\mathcal K_{\rm analytic}-\mathcal K_{\rm quad}|}{|\mathcal K_{\rm analytic}|}
=5.25\times 10^{-16},
\]

and

\[
\max \frac{|\mathcal K_{\rm erf}-\mathcal K_{\rm Dawson}|}{|\mathcal K_{\rm erf}|}
=9.30\times 10^{-17}.
\]

So the analytic two-pulse Dawson kernel is numerically correct at double-precision roundoff for this test. This is a kernel-level validation, not a full TDSE or dipole-converged atomic calculation.
