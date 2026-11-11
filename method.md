# Method Notes - Off-Axis Freeform Illumination Lens Design

## 1. Problem setting

A uniform collimated beam with irradiance $I_s$ illuminates a lens whose front
surface is planar and whose back surface is freeform. The front surface lies in
the plane $z=0$ and covers a rectangular domain $D_s$. The incident rays travel
along $+z$ and enter the lens normally, so they are not deflected at the front
surface. Refraction happens only at the freeform back surface.

The back surface is represented as a height field over the front-surface
coordinates,

$$
\mathbf{q}(x,y) = \left( x, y, z_\theta(x,y) \right),
$$

where $z_\theta$ is a scalar function (in the implementation, a B-spline field).
The projection plane is at $z=d$, and the prescribed target irradiance $E_t$ is
defined on a rectangular domain $D_t$ in that plane.

Let $n_{\mathrm{lens}}$ and $n_{\mathrm{air}}$ be the refractive indices of the
lens and the surrounding medium, and let
$\eta_{\mathrm{lens}}=n_{\mathrm{lens}}/n_{\mathrm{air}}$. Snell's law at the
back surface is used in the usual vector form. For Fresnel losses, write
$\eta=n_{\mathrm{in}}/n_{\mathrm{out}}$ (the ratio of the incident and outgoing
indices, so that
$\sin\theta_{\mathrm{out}}=\eta\sin\theta_{\mathrm{in}}$). For unpolarized
light the reflectance is

$$
r_s=\frac{\eta\cos\theta_{\mathrm{in}}-\cos\theta_{\mathrm{out}}}
         {\eta\cos\theta_{\mathrm{in}}+\cos\theta_{\mathrm{out}}},
\qquad
r_p=\frac{\cos\theta_{\mathrm{in}}-\eta\cos\theta_{\mathrm{out}}}
         {\cos\theta_{\mathrm{in}}+\eta\cos\theta_{\mathrm{out}}},
\qquad
R=\frac{r_s^2+r_p^2}{2}.
$$

The transmitted fraction is $\tau=1-R$ (per interface, ignoring absorption).

In the continuous model, let $\phi:D_s\to D_t$ be the ray map induced by the
lens. Energy conservation, including the accumulated transmission factor
$L_{\mathrm{ref}}(\mathbf{p})$, reads

$$
I_s(\mathbf{p}) L_{\mathrm{ref}}(\mathbf{p}) =
E_t\left(\phi(\mathbf{p})\right)
\left|\det\nabla\phi(\mathbf{p})\right|.
$$

## 2. Pipeline overview

The design is computed in three stages:

1. **Reflection-aware optimal-transport initialization.** A semi-discrete
   optimal transport (OT) problem gives an energy-consistent ray
   correspondence. A first OT solve ignores Fresnel loss; a preliminary lens
   fitted from it is used to estimate incidence angles and lens thickness, and
   a second, Fresnel-aware fixed-point solve corrects the transport masses.
2. **Surface fitting with correspondence refinement.** A B-spline height field
   is fitted to the target normals implied by Snell's law. Because an OT map is
   not exactly integrable at large deflection angles, the correspondence is
   refined on a triangulated representation by jointly penalizing irradiance
   error, curl, and mesh distortion, alternating with surface fitting.
3. **Differentiable ray tracing.** The lens geometry is refined end-to-end by
   tracing sampled rays, applying Fresnel transmission, splatting the rays onto
   the target image, and minimizing the relative RMS error (RRMSE) against the
   target.

## 3. Stage 1 - Reflection-aware OT initialization

### 3.1 Semi-discrete OT

The quadratic transport cost between the source measure $I_s$ and the target
measure $E_t$ is

$$
d(I_s,E_t)=\inf_{\phi\in\mathcal{M}}
\int_{D_s}\left\|\phi(\mathbf{p})-\mathbf{p}\right\|^2
\mathrm{d}I_s(\mathbf{p}),
$$

where $\mathcal{M}$ is the set of admissible energy-conserving maps.

The front-surface domain is sampled with $N$ points
$S_0=\lbrace \mathbf{p}_i \rbrace_{i=1}^{N}$, including the four corners and
covering the interior and boundary. A Voronoi tessellation of $D_s$ gives each
sample a transport mass

$$
\lambda_i=\int_{V_i} I_s(\mathbf{p}) \mathrm{d}\mathbf{p}.
$$

The target domain is assumed to be geometrically similar to $D_s$; a pattern
that does not fill the domain can be embedded in a larger zero-irradiance
region. Initial target sites $\mathbf{s}_i$ are obtained by applying the same
similarity transform to $\mathbf{p}_i$. Together with the masses, they define
the discrete target measure

$$
\nu_0=\sum_{i=1}^{N}\lambda_i \delta_{\mathbf{s}_i}.
$$

The target irradiance $E_t$, normalized to the same total mass, is an
absolutely continuous measure on $D_t$. The semi-discrete OT problem between
the continuous target measure and $\nu_0$ is solved for scalar weights $w_i$
that define a weighted Voronoi (power) diagram on $D_t$:

$$
\mathrm{Vor}_i^{w} =
\left\lbrace
\mathbf{x}\in D_t :
\lVert \mathbf{x}-\mathbf{s}_i \rVert^2-w_i
\le
\lVert \mathbf{x}-\mathbf{s}_j \rVert^2-w_j
\quad \forall j
\right\rbrace.
$$

The weights minimize the dual objective

$$
F(w) =
-\sum_i \lambda_i w_i -
\sum_i \int_{\mathrm{Vor}_i^{w}}
\left(
\lVert \mathbf{x}-\mathbf{s}_i \rVert^2-w_i
\right)
\mathrm{d}\mu(\mathbf{x}),
$$

where $\mu$ is the target measure with density $E_t$. At the solution each
cell carries its prescribed mass,

$$
\mu\left(\mathrm{Vor}_i^{w}\right)=\lambda_i,
$$

and the map sending every point of cell $i$ to $\mathbf{s}_i$ is the quadratic
OT map. In practice, the correspondence used by the lens pipeline pairs each
source sample $\mathbf{p}_i$ with the irradiance-weighted centroid of its cell,

$$
\bar{\mathbf{q}}_i =
\frac{
\int_{\mathrm{Vor}_i^{w}}
\mathbf{q} E_t(\mathbf{q}) \mathrm{d}\mathbf{q}
}{
\int_{\mathrm{Vor}_i^{w}}
E_t(\mathbf{q}) \mathrm{d}\mathbf{q}
}.
$$

The pairs $(\mathbf{p}_i,\bar{\mathbf{q}}_i)$ form the initial ray
correspondence. The dual is convex and is solved with a standard convex
optimizer.

### 3.2 Preliminary lens

The conventional OT correspondence is first used to fit a preliminary B-spline
lens with the surface-fitting procedure of Section 4.2. At large deflection
angles this lens only approximates the desired mapping, but it provides a
useful estimate of the lens thickness profile. That estimate matters because
the thickness affects the exit point, the incidence angle at the freeform
surface, and therefore the Fresnel loss of each ray.

### 3.3 Fresnel-aware mass correction

Starting from the converged weights of the conventional OT solve, the
reflection-aware solve keeps the sites $\mathbf{s}_i$ fixed and iterates the
following steps.

**Step 1.** Build the power diagram for the current weights $w^k$ and compute
the irradiance-weighted centroid $\mathbf{q}_i^k$ of each cell $P_i^{w^k}$.

**Step 2.** For sample $\mathbf{p}_i$, form a hypothetical ray path: enter the
preliminary lens at $\mathbf{p}_i$, leave the back surface at the point
predicted by the current lens model, and travel to $\mathbf{q}_i^k$. This path
is used only to estimate the Fresnel loss; it does not have to satisfy Snell's
law exactly.

**Step 3.** Accumulate the transmission factor $\tau_i^k$ along that path and
correct the discrete mass,

$$
\hat{\lambda}_i^{k} =
\frac{\tau_i^{k} \lambda_i}
     {\sum_j \tau_j^{k} \lambda_j}.
$$

Samples with larger reflection loss thus receive less transmitted energy.

**Step 4.** Update the weights by a fixed-point step on the mass mismatch,

$$
w_i^{k+1} =
w_i^{k} -
\alpha_r
\left(
\int_{P_i^{w^k}} E_t(\mathbf{q}) \mathrm{d}\mathbf{q} -
\hat{\lambda}_i^{k}
\right),
\qquad \alpha_r\gt0.
$$

In the implementation the step size is set to
$\alpha_r=|D_t|/10$, where $|D_t|$ is the area of the projection domain.
Anderson acceleration is used to improve convergence. After convergence, the
cell centroids $\mathbf{q}_i$ are collected as the target samples $D_0$, and the
final initial correspondence pairs $S_0$ with $D_0$.

## 4. Stage 2 - Surface fitting and correspondence refinement

### 4.1 B-spline surface and target normals

The back surface is a cubic B-spline scalar field $z_\theta$ on a uniform
$n\times n$ tensor-product knot grid with open uniform knot vectors in both
parametric directions. Its (unnormalized) normal is

$$
\mathbf{n}_b(x,y) =
\left(
-\frac{\partial z_\theta}{\partial x},
-\frac{\partial z_\theta}{\partial y},
1
\right).
$$

Given a source point $\mathbf{p}(x,y)$ and its target $\mathbf{t}(x,y)$, the
incoming direction inside the lens is $(0,0,1)^\top$ and the desired outgoing
direction is proportional to $\mathbf{t}-\mathbf{q}$. Snell's law in vector
form gives the target normal

$$
\mathbf{n}_b
\propto
\eta_{\mathrm{lens}} (0,0,1)^\top -
\frac{\mathbf{t}(x,y)-\mathbf{q}(x,y)}
     {\lVert \mathbf{t}(x,y)-\mathbf{q}(x,y) \rVert_2}.
$$

Because the $z$-component of $\mathbf{n}_b$ is positive, the target gradient of
the height field is obtained by normalizing with $n_3$:

$$
(g_1,g_2) =
\left(
-\frac{n_1}{n_3},
-\frac{n_2}{n_3}
\right).
$$

### 4.2 Surface-fitting objective

The spline is optimized so that its gradient matches the target gradient at a
set of sample points $\mathcal{S}$:

$$
\mathcal{L}_{\mathrm{grad}} =
\sum_{\mathbf{p}\in\mathcal{S}}
\left\lVert
\left(
\frac{\partial z_\theta}{\partial x}(\mathbf{p}),
\frac{\partial z_\theta}{\partial y}(\mathbf{p})
\right) -
\left(g_1(\mathbf{p}),g_2(\mathbf{p})\right)
\right\rVert_2.
$$

Two regularizers keep the lens physically plausible. The mean curvature $H$ is
penalized on a regular sample set $\mathcal{S}_{\mathrm{reg}}$ taken at the
centers of the $n\times n$ spline cells:

$$
\mathcal{L}_{\mathrm{curv}} =
\frac{1}{|\mathcal{S}_{\mathrm{reg}}|}
\sum_{\mathbf{p}\in\mathcal{S}_{\mathrm{reg}}}
|H(\mathbf{p})|.
$$

For thickness, a desired minimum thickness $h$ is prescribed. The target
average height is computed with a stop-gradient,

$$
\bar{h} =
\mathrm{sg}\left(
\frac{1}{|\mathcal{S}_{\mathrm{reg}}|}
\sum_{\mathbf{p}\in\mathcal{S}_{\mathrm{reg}}} z_\theta(\mathbf{p}) -
\min_{\mathbf{p}\in\mathcal{S}_{\mathrm{reg}}} z_\theta(\mathbf{p}) +
h
\right),
$$

and the average height is matched to it:

$$
\mathcal{L}_{\mathrm{thic}} =
\left(
\frac{1}{|\mathcal{S}_{\mathrm{reg}}|}
\sum_{\mathbf{p}\in\mathcal{S}_{\mathrm{reg}}} z_\theta(\mathbf{p}) -
\bar{h}
\right)^2.
$$

The surface-fitting loss is

$$
\mathcal{L}_1 =
\lambda_{\mathrm{grad}}\mathcal{L}_{\mathrm{grad}} +
\lambda_{\mathrm{curv}}\mathcal{L}_{\mathrm{curv}} +
\lambda_{\mathrm{thic}}\mathcal{L}_{\mathrm{thic}}.
$$

In the first round, $\mathcal{S}$ and the target points are the OT samples
$S_0$ and $D_0$. In later rounds they are replaced by the refined
correspondence from Section 4.3.

### 4.3 Triangulated correspondence

The initial samples $S_0$ are Delaunay-triangulated to form a mesh $T$ on the
front surface. Using the same connectivity, the paired target samples form a
mesh $T'$ on the projection plane. The connectivity is fixed during
optimization; when a finer discretization is needed, each triangle is
uniformly subdivided into four (Loop subdivision), and the new vertices become
additional optimization variables.

The optimization variables are the front-surface vertices
$S=\lbrace(\mathbf{x}_i,\mathbf{y}_i)\rbrace$ and the target vertices
$D=\lbrace(\mathbf{u}_i,\mathbf{v}_i)\rbrace$. Boundary vertices are
constrained to move along the boundary, and corner vertices are fixed. The lens
geometry $z_\theta$ is held fixed during a correspondence-refinement step.


### 4.4 Irradiance matching

The incident flux of a front-surface triangle $t_i$ is proportional to its area
$A(t_i)$. To account for Fresnel loss, the raw flux is corrected by an
estimated total transmittance,

$$
\tau_{\mathrm{tot}}=\tau_s \tau_b,
$$

where $\tau_s$ is the front-surface transmittance and $\tau_b$ is the
back-surface transmittance. The latter is estimated from the target normal
implied by the current correspondence rather than from the actual surface
normal. This ideal-path estimate keeps the early optimization stable when
surface normals are poor or when a ray would otherwise undergo total internal
reflection. The transmittance of triangle $t_i$ is the average of its three
vertex values, denoted $\bar{\tau}_{t_i}$, and the corrected transmitted flux
is

$$
\ell^c_{t_i}=\bar{\tau}_{t_i}\ell_{t_i}.
$$

The flux assigned to the corresponding projection triangle is normalized so
that the total transmitted energy matches the target:

$$
\ell'_{t_i} =
c_{\mathrm{total}}
\frac{\ell^c_{t_i}}{\sum_j \ell^c_{t_j}},
$$

where $c_{\mathrm{total}}$ is the total flux required by the target irradiance
distribution. The projection domain is divided into pixels; assuming uniform
irradiance inside each projection triangle, the flux contribution to pixel
$p_j$ is

$$
\ell_{t'_i,p_j} =
\ell'_{t_i}
\frac{A(t'_i\cap p_j)}{A(t'_i)},
\qquad
\ell_{p_j} =
\sum_i \ell_{t'_i,p_j}.
$$

With target pixel values $\ell_{p_j}^{\mathrm{target}}$ and
$\ell_{\max}=\max_j \ell_{p_j}^{\mathrm{target}}$ the pixel count $m$, the
irradiance loss is

$$
\mathcal{L}_{\mathrm{irra}} =
\frac{1}{m}
\sum_{j=1}^{m}
\left(
\frac{\ell_{p_j}-\ell_{p_j}^{\mathrm{target}}}{\ell_{\max}}
\right)^2.
$$

A boundary term keeps the target vertices inside $D_t$:

$$
\mathcal{L}_{\mathrm{bdr}} =
\sum_{\mathbf{q}_i\in D}
\mathrm{dist}(\mathbf{q}_i,D_t)^2.
$$

### 4.5 Integrability

A correspondence is physically realizable only if the induced target gradient
field $\mathbf{g}=(g_1,g_2)$ is integrable. On the triangulated mesh this is
enforced approximately by penalizing the discrete curl on every triangle.
Linearly interpolating $\mathbf{g}$ inside $t_i$, the curl is computed as the
boundary line integral

$$
\mathrm{curl}_{t_i} =
\oint_{\partial t_i}\mathbf{g}\cdot \mathrm{d}\mathbf{l} =
\sum_{k=0}^{2}
\frac{\mathbf{g}(\mathbf{v}_k)+\mathbf{g}(\mathbf{v}_{(k+1)\bmod 3})}{2}
\cdot
\left(\mathbf{v}_{(k+1)\bmod 3}-\mathbf{v}_k\right),
$$

where $\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2$ are the triangle's vertices. The
integrability loss is

$$
\mathcal{L}_{\mathrm{int}} =
\frac{1}{|D_s|}
\sum_{t_i\in T}
\left(\mathrm{curl}_{t_i}\right)^2.
$$

### 4.6 Mesh regularization

Triangle areas on the front mesh are kept within a range $[\alpha,\beta]$ with
smooth barriers. Let
$0\lt\alpha\lt\tilde{\alpha}\lt\tilde{\beta}\lt\beta$. The two barriers are

$$
B_{\mathrm{lower}}(t_i)=
\begin{cases}
+\infty, & A(t_i)\lt\alpha,\\
\left(\frac{\tilde{\alpha}-\alpha}{A(t_i)-\alpha}\right)^2-1,
& \alpha\lt A(t_i)\le\tilde{\alpha},\\
0, & A(t_i)\gt\tilde{\alpha}.
\end{cases}
$$

$$
B_{\mathrm{upper}}(t_i)=
\begin{cases}
+\infty, & A(t_i)\gt\beta,\\
\left(\frac{\beta-\tilde{\beta}}{\beta-A(t_i)}\right)^2-1,
& \tilde{\beta}\le A(t_i)\lt\beta,\\
0, & A(t_i)\lt\tilde{\beta}.
\end{cases}
$$

Large triangles are allowed on the projection mesh (for example in dim
regions, where few rays are needed), so $T'$ receives only a lower barrier

$$
B'_{\mathrm{lower}}(t'_i)=
\begin{cases}
+\infty, & A(t'_i)\lt\alpha',\\
\left(\frac{\tilde{\alpha}'-\alpha'}{A(t'_i)-\alpha'}\right)^2-1,
& \alpha'\lt A(t'_i)\le\tilde{\alpha}',\\
0, & A(t'_i)\gt\tilde{\alpha}'.
\end{cases}
$$

The thresholds are expressed as fractions of the reference average triangle
area of the current mesh and are kept fixed during a refinement round. In the
implementation they are set to $1\%$, $20\%$, $150\%$, $200\%$ of the reference
front-surface area for $\alpha$, $\tilde{\alpha}$, $\tilde{\beta}$, $\beta$,
and to $0$ and $2\%$ of the reference projected area for $\alpha'$ and
$\tilde{\alpha}'$. The regularization loss is

$$
\mathcal{L}_{\mathrm{reg}} =
\sum_{t_i\in T}
\left(B_{\mathrm{lower}}(t_i)+B_{\mathrm{upper}}(t_i)\right) +
\sum_{t'_i\in T'} B'_{\mathrm{lower}}(t'_i).
$$

Before the main refinement, a short preprocessing step fixes flipped or
degenerate projection triangles by optimizing only the vertices of $T'$:

$$
\mathcal{L}_{\mathrm{pre}} =
\sum_{t'_i\in T',\; A(t'_i)\lt\tilde{\alpha}'}
\left(\tilde{\alpha}'-A(t'_i)\right)^2,
$$

stopping once every projection triangle satisfies
$A(t'_i)\gt\frac{1}{2}\tilde{\alpha}'$.

### 4.7 Alternating optimization

Correspondence refinement and surface fitting alternate. With the lens fixed,
the correspondence minimizes

$$
\mathcal{L}_2 =
\lambda_{\mathrm{irra}}\mathcal{L}_{\mathrm{irra}} +
\lambda_{\mathrm{int}}\mathcal{L}_{\mathrm{int}} +
\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}} +
\lambda_{\mathrm{bdr}}\mathcal{L}_{\mathrm{bdr}}.
$$

The weights balance irradiance accuracy, integrability, mesh quality, and the
boundary constraint. The values used in the implementation are
$\lambda_{\mathrm{irra}}=1$, $\lambda_{\mathrm{int}}=3\times10^{5}$,
$\lambda_{\mathrm{reg}}=1\times10^{-9}$ and
$\lambda_{\mathrm{bdr}}=1\times10^{-1}$. The integrability weight is ramped up
over the first refinement round to avoid an excessively large gradient at the
beginning.

After each refinement, the pairs $(S,D)$ become the samples and targets for the
next surface-fitting round. The process is repeated coarse-to-fine. In the
third refinement round the triangulation is subdivided (each triangle into
four), and the B-spline grid is refined.

## 5. Stage 3 - Differentiable ray tracing

The lens from Stage 2 is close to the target but still carries fitting error
and curvature-induced redistribution. A final end-to-end refinement traces many
rays through the actual surface.

The front surface is sampled uniformly with $N_r$ rays
$\mathcal{S}_{\mathrm{ray}}=\lbrace\mathbf{s}_k\rbrace$. Because the source is
a uniform collimated beam, all rays carry equal incident weight. Each ray is
traced through the lens with Snell's law, and the accumulated Fresnel
transmittance $\tau_k$ is computed along its path. The transmitted energies are
normalized so that their total matches the target flux:

$$
\tilde{\beta}_k =
c_{\mathrm{total}}
\frac{\beta_k}{\sum_h \beta_h},
\qquad
\beta_k\propto\tau_k.
$$

A ray intersects the projection plane at $\mathbf{y}_k$. To obtain a
differentiable image, its energy is distributed over a local pixel window with
a Gaussian kernel. Let $\mathbf{c}_j$ be the center of pixel $P_j$, and let
$\tilde{\mathcal{N}}_k$ be the full window around $\mathbf{y}_k$. For each
pixel in the window,

$$
G_{kj} =
\exp\left(
-\frac{\lVert \mathbf{y}_k-\mathbf{c}_j \rVert^2}{2\sigma^2}
\right),
\qquad
\omega_{kj} =
\frac{G_{kj}}{\sum_{h\in\tilde{\mathcal{N}}_k}G_{kh}}.
$$

Only contributions to valid pixels
$j\in\mathcal{N}_k\subseteq\tilde{\mathcal{N}}_k$ are accumulated:

$$
\ell_{P_j}^{\mathrm{sim}} =
\sum_{k=1}^{N_r}\tilde{\beta}_k \omega_{kj},
\qquad
\omega_{kj}=0 \quad \text{if } P_j\notin\mathcal{N}_k.
$$

The simulated image is compared with the target using RRMSE:

$$
\mathcal{L}_{\mathrm{sim}} =
\sqrt{
\frac{
\sum_{j=1}^{m}
\left(\ell_{P_j}^{\mathrm{sim}}-\ell_{P_j}\right)^2
}{
\sum_{j=1}^{m}\ell_{P_j}^2
}
}.
$$

The final objective also includes the curvature and thickness regularizers:

$$
\mathcal{L}_3 =
\lambda_{\mathrm{sim}}\mathcal{L}_{\mathrm{sim}} +
\lambda_{\mathrm{curv}}\mathcal{L}_{\mathrm{curv}} +
\lambda_{\mathrm{thic}}\mathcal{L}_{\mathrm{thic}}.
$$

The splatting window radius and kernel width are implementation constants. In
the reference experiments $r=2$ and $\sigma=0.5$, and the three loss weights
are $\lambda_{\mathrm{sim}}=\lambda_{\mathrm{curv}}=\lambda_{\mathrm{thic}}=1$.

## 6. Known limitations

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
