
# Method

We design a single-freeform-surface illumination lens that transforms a uniform collimated beam into a prescribed irradiance pattern, including under strongly off-axis illumination conditions. The method combines **Fresnel-loss-aware optimal transport**, **alternating ray-mapping and surface optimization**, and **end-to-end differentiable ray tracing**.

## 1. Problem formulation

The lens has a planar entrance surface at $z=0$ and a freeform exit surface. Incident rays travel in the positive $z$-direction and enter the lens normally. Let $D_{\mathrm{s}}\subset\mathbb{R}^2$ be the entrance aperture and $D_{\mathrm{t}}\subset\mathbb{R}^2$ the target domain on the projection plane. The source irradiance is $I_s$, and the desired target irradiance is $E_t$, represented by a grayscale image.

The lens induces a ray mapping $\phi:D_{\mathrm{s}}\rightarrow D_{\mathrm{t}}$. Accounting for Fresnel reflection losses, energy conservation requires

```math
I_s(p_s)\,L_{\mathrm{ref}}(p_s)
=
E_t\bigl(\phi(p_s)\bigr)
\left|\det\bigl(\nabla\phi(p_s)\bigr)\right|,
```

where $L_{\mathrm{ref}}(p_s)$ is the accumulated transmission factor along the ray originating at $p_s$. Refraction follows Snell's law, while the Fresnel reflectance $R$ determines the interface transmittance $\tau=1-R$ (with accumulated transmittance obtained from the interfaces along the ray path).

Our pipeline has three stages:

1. Construct an initial, energy-consistent ray mapping using reflection-aware semi-discrete optimal transport (OT).
2. Alternately optimize the ray mapping and a B-spline freeform surface so that the mapping is both irradiance-preserving and physically realizable.
3. Refine the resulting lens directly using differentiable ray tracing against the target image.

## 2. Reflection-aware ray-mapping initialization

We sample the entrance aperture to obtain $\mathcal{S}_0=\{p_i\}_{i=1}^N$. Each sample receives an incident-energy mass from its source-domain Voronoi cell $V_i$:

```math
\lambda_i=\int_{V_i}I_s(p)\,dp.
```

For the OT computation, the source and target distributions are normalized to equal total mass. We initialize target-domain transport sites $\overline{\mathcal{S}}_0=\{s_i\}_{i=1}^N$ using a similarity transformation of the entrance samples. Conventional semi-discrete OT adjusts the weights $w_i$ of target-domain **power cells** $P_i^w$ so that

```math
\int_{P_i^w} E_t(q)\,dq=\lambda_i.
```

The irradiance-weighted centroid of each cell defines its corresponding target point. Pairing these centroids with the source samples gives an initial mapping, from which we fit a **provisional lens**. This provisional surface provides an estimate of lens thickness and, hence, the relevant ray directions and Fresnel losses.

To account for reflection, we iteratively re-estimate the transmittance $\tau_i^k$ of a hypothetical ray joining source sample $p_i$ to the current target-cell centroid through the provisional lens. Its effective transported mass becomes

```math
\widehat{\lambda}_i^k
=
\frac{\tau_i^k\lambda_i}{\sum_{j=1}^{N}\tau_j^k\lambda_j}.
```

Because these masses depend on the current power cells, we update the OT weights using a fixed-point iteration:

```math
w_i^{k+1}
=
w_i^k-\alpha_r\left(
\int_{P_i^{w^k}} E_t(q)\,dq
-\widehat{\lambda}_i^k
\right).
```

The converged power-cell centroids form the target samples $\mathcal{D}_0=\{q_i\}_{i=1}^N$. The pairs $(p_i,q_i)$ define a **reflection-aware initial correspondence**, compensating for spatially varying transmission losses while respecting the target energy distribution.

## 3. Alternating ray-mapping and lens-surface optimization

A transport map may satisfy energy conservation without being realizable by a single continuous refracting surface. We therefore alternate between refining the mapping **with the lens fixed** and fitting the lens **with the mapping fixed**.

### 3.1 B-spline surface fitting

We represent the freeform exit surface as a differentiable B-spline height field $z_\theta$:

```math
\mathbf{q}(x,y)=(x,y,z_\theta(x,y)),
\qquad
\mathbf{n}_b(x,y)=\left(-\frac{\partial z_\theta}{\partial x},-\frac{\partial z_\theta}{\partial y},1\right).
```

Here, $\mathbf{q}(x,y)$ is the **3D exit-surface point**, distinct from the **2D target sample** $q_i$ used in OT. For the prescribed 3D target point $\mathbf{t}(x,y)$, Snell's law gives a target normal direction

```math
\overline{\mathbf{n}}_b(x,y)
=
\eta_{\mathrm{lens}}\begin{pmatrix}0\\0\\1\end{pmatrix}
-
\frac{\mathbf{t}(x,y)-\mathbf{q}(x,y)}
{\|\mathbf{t}(x,y)-\mathbf{q}(x,y)\|_2},
```

where $\eta_{\mathrm{lens}}$ is the refractive-index ratio of lens material to air. If $\overline{\mathbf{n}}_b=(\overline{n}_1,\overline{n}_2,\overline{n}_3)$, the target surface gradient is

```math
\overline{\mathbf{g}}
=(\overline{g}_1,\overline{g}_2)
=
\left(-\frac{\overline{n}_1}{\overline{n}_3},
-\frac{\overline{n}_2}{\overline{n}_3}\right).
```

We fit the B-spline surface by minimizing the discrepancy between its gradient and $\overline{\mathbf{g}}$ at the current entrance samples $\mathcal{S}$, while regularizing curvature and thickness:

```math
\begin{aligned}
\mathcal{L}_{\mathrm{grad}}
&=\sum_{p\in\mathcal{S}}
\left\|\nabla z_\theta(p)-\overline{\mathbf{g}}(p)\right\|_2,\\
\mathcal{L}_1
&=\lambda_{\mathrm{grad}}\mathcal{L}_{\mathrm{grad}}
+\lambda_{\mathrm{curv}}\mathcal{L}_{\mathrm{curv}}
+\lambda_{\mathrm{thic}}\mathcal{L}_{\mathrm{thic}}.
\end{aligned}
```

### 3.2 Ray-mapping refinement

We triangulate the entrance samples into $\mathcal{T}=\{t_i\}$ and connect their corresponding target samples with the **same connectivity** to form $\mathcal{T}'=\{t_i'\}$. Each source triangle $t_i$ maps to one target triangle $t_i'$. Let $\mathcal{S}=\{(x_i,y_i)\}$ and $\mathcal{D}=\{(u_i,v_i)\}$ denote the current source and target vertices. The optimization updates these vertex positions while the lens geometry and triangle connectivity remain fixed.

**Irradiance consistency.** For each source triangle, we estimate the total transmittance from the intended ray directions and their Snell-derived target normals, rather than relying on potentially inaccurate surface normals early in the process, which also avoids unstable total-internal-reflection evaluations. Denoting its average transmittance by $\overline{\tau}_{t_i}$, we assign its mapped target triangle the flux

```math
l_{t_i}'
=c_{\mathrm{total}}
\frac{\overline{\tau}_{t_i}\,\mu(t_i)}
{\sum_{t_j\in\mathcal{T}}\overline{\tau}_{t_j}\,\mu(t_j)},
```

where $\mu(t_i)$ is the source-triangle area and $c_{\mathrm{total}}$ is the total flux prescribed by the target image. The normalization places the simulated and desired images on the same total-flux scale; it does not represent a prediction of absolute optical throughput.

Assuming uniform irradiance within each mapped triangle, the flux assigned to target pixel $p_j$ is computed from triangle–pixel overlap:

```math
l_{p_j}
=\sum_{t_i'\in\mathcal{T}'}
l_{t_i}'\frac{A(t_i'\cap p_j)}{A(t_i')},
```

where $A(\cdot)$ denotes planar area. An irradiance loss $\mathcal{L}_{\mathrm{irra}}$ penalizes the difference between these pixel values $l_{p_j}$ and the target values $\overline{l}_{p_j}$.

**Integrability.** A continuous height field requires the target gradient $\overline{\mathbf{g}}$ to be curl-free. We measure discrete curl over each source triangle by the boundary integral

```math
\mathrm{curl}_{t_i}
=\oint_{\partial t_i}\overline{\mathbf{g}}\cdot d\mathbf{l},
\qquad
\mathcal{L}_{\mathrm{int}}
=\frac{1}{|D_{\mathrm{s}}|}
\sum_{t_i\in\mathcal{T}}(\mathrm{curl}_{t_i})^2.
```

**Geometric validity.** Mesh regularization $\mathcal{L}_{\mathrm{reg}}$ prevents flipped or degenerate triangles and keeps entrance sampling well distributed. A boundary loss $\mathcal{L}_{\mathrm{bdr}}$ discourages target vertices from leaving $D_{\mathrm{t}}$.

The complete mapping-refinement objective is

```math
\mathcal{L}_2
=\lambda_{\mathrm{irra}}\mathcal{L}_{\mathrm{irra}}
+\lambda_{\mathrm{int}}\mathcal{L}_{\mathrm{int}}
+\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}
+\lambda_{\mathrm{bdr}}\mathcal{L}_{\mathrm{bdr}}.
```

We then refit the B-spline surface to the updated correspondence using $\mathcal{L}_1$, and repeat the two steps to jointly improve irradiance fidelity and physical realizability.

## 4. End-to-end differentiable ray-tracing refinement

The triangle-based mapping model is an approximation: real rays do not necessarily land uniformly inside their assigned target triangles, and spline fitting introduces additional residual errors. We therefore optimize the lens geometry directly using differentiable ray tracing.

We sample rays $\mathcal{S}_{\mathrm{ray}}=\{\mathbf{s}_k\}_{k=1}^{N_r}$ on the entrance aperture, trace them through the current lens using Snell's law, and evaluate the accumulated Fresnel transmittance $\tau_k$. Since the incident beam is uniform, each ray has equal incident weight, so we set its relative transmitted energy to $\beta_k=\tau_k$. The normalized energy is

```math
\widetilde{\beta}_k
=c_{\mathrm{total}}\frac{\beta_k}{\sum_{h=1}^{N_r}\beta_h}.
```

Let $\mathbf{y}_k$ be the ray's intersection with the projection plane, and $\mathbf{c}_j$ the center of pixel $P_j$. A Gaussian kernel distributes the ray energy to nearby pixels. If $\widetilde{\mathcal{N}}_k$ is the complete local pixel window, we use

```math
\omega_{kj}
=\frac{\exp\left(-\|\mathbf{y}_k-\mathbf{c}_j\|_2^2/(2\sigma^2)\right)}
{\sum_{h\in\widetilde{\mathcal{N}}_k}
\exp\left(-\|\mathbf{y}_k-\mathbf{c}_h\|_2^2/(2\sigma^2)\right)}.
```

The kernel is normalized over the **full** window; only contributions to valid pixels inside the image are accumulated:

```math
l_{P_j}^{\mathrm{sim}}
=\sum_{k=1}^{N_r}\widetilde{\beta}_k\,\omega_{kj},
```

with $\omega_{kj}=0$ when $P_j$ is outside the valid local neighborhood of ray $k$. We compare the rendered image with the desired pixel values $\overline{l}_{P_j}$ using relative root-mean-square error:

```math
\mathcal{L}_{\mathrm{sim}}
=\sqrt{\frac{\sum_j(l_{P_j}^{\mathrm{sim}}-\overline{l}_{P_j})^2}
{\sum_j\overline{l}_{P_j}^{\,2}}}.
```

Finally, we optimize the B-spline parameters $\theta$ using

```math
\mathcal{L}_3
=\lambda_{\mathrm{sim}}\mathcal{L}_{\mathrm{sim}}
+\lambda_{\mathrm{curv}}\mathcal{L}_{\mathrm{curv}}
+\lambda_{\mathrm{thic}}\mathcal{L}_{\mathrm{thic}}.
```

This final stage corrects errors that the discrete ray-mapping approximation and gradient-based surface reconstruction cannot fully resolve.

## Known limitations

- The formulation assumes a uniform collimated source and a single freeform
  refracting surface. Extended sources and multi-surface systems are not
  implemented.
- The reference experiments use one off-axis configuration. Angle sweeps,
  tolerance analysis, and manufacturability constraints are not included.
- Near the critical angle the Fresnel reflectance is large. The rendered
  images are normalized to the target total flux, so they show the relative
  irradiance distribution rather than absolute efficiency.
- Total internal reflection is avoided during optimization by estimating the
  back-surface transmittance from target normals. The final designs stay below
  the critical deflection; TIR is not modeled as an explicitly differentiable
  event.
- Mesh connectivity is fixed after the initial Delaunay triangulation. A
  different topology requires re-meshing and re-running the pipeline.
- This is research code. It is memory-intensive and has not been optimized for
  production use.
