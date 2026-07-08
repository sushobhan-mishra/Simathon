"""
GALAXY COLLISION -- Real Gravitational N-Body Physics
Built with Python + Taichi

CONTROLS
--------
  LMB Hold         : Gravitational attractor (pull stars toward cursor)
  RMB Hold         : Repulsor burst (push stars away from cursor)
  SPACE            : Pause / Resume
  T                : Toggle particle trails (expensive -- off by default)
  F                : Toggle spiral field-line overlay
  G                : Toggle glow halos on galactic nuclei
  N                : Toggle density-based nebula cloud (artistic)
  M                : Toggle automatic cinematic camera
  H                : Toggle procedural Milky Way background
  J                : Toggle parallax stars
  K                : Toggle dust particles
  Y                : Toggle shockwave rings around the nuclei
  U                : Toggle tidal streamers
  I                : Toggle lens flares
  C                : Toggle cinematic color grading
  E                : Toggle orbital-energy colour diagnostic
  B                : Toggle two-pass star bloom rendering
  P                : Toggle performance mode (fewer stars, minimal FX)
  +  or  =         : Speed up simulation (x1.25)
  -                : Slow down simulation (x0.80)
  R                : Full reset to initial conditions
  1                : Spawn a dwarf galaxy at cursor (one-time)
  Q                : Quit

Performance tips (i3 / integrated GPU)
---------------------------------------
  * Press P to toggle performance mode (6k stars, no trails, smaller glow)
  * Trails (T) are OFF by default -- turn on only if FPS is comfortable
  * Lower star counts are set automatically on CPU fallback

Physics
-------
  Stars per galaxy : 6,000  (12,000 total)  [perf-tuned for i3]
  Integration      : Semi-implicit Euler (v update THEN p update)
  Gravity law      : a = G*M / (r^2 + eps^2)^(3/2)  [Plummer softening]
  Nuclei           : Super-massive black holes, mutually attracted
  Tidal heating    : Stars whiten/redden during close nuclear passage
  Boundary         : Escaped stars respawn on circular orbits near
                     nearest nucleus -- screen stays alive forever
"""

import taichi as ti
import math
import time
import numpy as np

# ---------------------------------------------------------------------------
#  Taichi init -- GPU first, CPU fallback
# ---------------------------------------------------------------------------
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    print("[Taichi] GPU backend initialised.")
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    print("[Taichi] CPU fallback -- reduce N_PER_GAL if FPS is low.")

# ---------------------------------------------------------------------------
#  PHYSICAL & RENDERING CONSTANTS
# ---------------------------------------------------------------------------

# Window -- 1280x800 is easier on integrated/low-end GPUs
WINDOW_W = 1280
WINDOW_H = 800

# Star counts -- reduced for i3 / Vulkan software rendering
# High-end GPU: raise N_PER_GAL to 18_000 for the full effect
N_PER_GAL = 6_000           # stars per galaxy  (12k total -- smooth on i3)
N_TOTAL = N_PER_GAL * 2   # two-galaxy total
MAX_N = N_PER_GAL * 3   # headroom: one optional dwarf galaxy

# Plummer gravitational softening -- prevents 1/r^2 singularity at r=0
# Force law: F = G*M*m / (r^2 + eps^2)^(3/2)
EPS = 5.0e-3           # softening length [sim units, 0..1 space]
EPS2 = EPS * EPS        # eps^2, precomputed

# Dimensionless gravitational constant (tuned so galaxies orbit ~1 crossing
# time before merging, giving time for tidal tails to develop)
G_CONST = 6.0e-5

# Galactic nucleus masses (in units where one star = 1 mass unit)
M_NUCLEUS = 8_000.0          # central SMBH equivalent
M_STAR = 1.0              # individual star mass (test particle)

# Disk structural parameters
R_DISK = 0.18           # exponential disk scale radius [sim units]
SPIRAL_ARMS = 3              # number of logarithmic spiral arms
WIND = 3.2            # winding tightness (larger = more wound)
DISK_THICK = 0.008          # vertical disk puffiness (projected scatter)

# Initial galaxy positions and bulk velocities
# Chosen for a grazing prograde encounter with pericentre ~0.06 sim units
GAL1_PX = 0.30
GAL1_PY = 0.50
GAL2_PX = 0.70
GAL2_PY = 0.50
GAL1_VX = 0.012
GAL1_VY = -0.006
GAL2_VX = -0.012
GAL2_VY = 0.006
GAL1_TILT = 0.3             # disk inclination (radians) -- relative to x-axis
GAL2_TILT = -0.8

# Integration
DT_BASE = 2.5e-4           # base timestep per sub-step

# Mouse interaction
MOUSE_G = 1.5e-5           # attractor gravity strength
MOUSE_R2 = 0.20 ** 2        # squared influence radius
REPULSOR_G = -8.0e-5         # repulsor strength (negative = push)

# ---------------------------------------------------------------------------
#  TAICHI FIELDS  -- ALL particle state lives here (never Python lists)
# ---------------------------------------------------------------------------
pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_N)  # x, y
vel = ti.Vector.field(2, dtype=ti.f32, shape=MAX_N)  # vx, vy
col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)  # r, g, b
alive = ti.field(dtype=ti.i32, shape=MAX_N)            # 1 = active

# Optional render-only color buffer. Physics and persistent star colours stay
# in col; this buffer is rebuilt only when cinematic grading is enabled.
grade_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)

# Optional render-only diagnostic: colour stars by local specific orbital
# energy relative to the nearest active nucleus. This never feeds back into
# forces or the persistent colour field.
energy_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)

# Optional render-only particle bloom buffer. It is derived from the active
# display colour each frame and drawn as a soft first pass before star cores.
star_soft_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)

# Galactic nuclei (up to 3: two main + one optional dwarf)
nuc_pos = ti.Vector.field(2, dtype=ti.f32, shape=3)
nuc_vel = ti.Vector.field(2, dtype=ti.f32, shape=3)
nuc_mass = ti.field(dtype=ti.f32, shape=3)
nuc_col = ti.Vector.field(3, dtype=ti.f32, shape=3)
nuc_alive = ti.field(dtype=ti.i32, shape=3)

# Static field-line spiral overlay
N_FIELD = 600              # halved for performance
fl_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_FIELD)
fl_col = ti.Vector.field(3, dtype=ti.f32, shape=N_FIELD)

# Glow halo points rebuilt each frame (slimmed -- fewer layers for i3)
MAX_GLOW = 400
glow_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_GLOW)
glow_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_GLOW)

# Single-element fields for drawing nucleus core dots via canvas.circles
nuc_dot_pos = ti.Vector.field(2, dtype=ti.f32, shape=3)
nuc_dot_col = ti.Vector.field(3, dtype=ti.f32, shape=3)

# Optional render-only camera view. This is a display transform only and does
# not alter the physical state or the gravitational dynamics.
view_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_N)
view_nuc_pos = ti.Vector.field(2, dtype=ti.f32, shape=3)

# Optional render-only nebula cloud: a sparse, density-inspired point cloud
# built from the current star distribution. It is artistic and does not feed
# back into any force calculation.
MAX_NEBULA = 1_800
nebula_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NEBULA)
nebula_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_NEBULA)

# Optional render-only Milky Way background. This is a static artistic layer
# and does not feed back into the physical simulation.
MAX_MILKY = 2_200
milky_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_MILKY)
milky_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_MILKY)

# Optional render-only parallax stars. These are static background points
# rendered at different scales to suggest depth without adding any physics.
MAX_PARALLAX = 1_600
parallax_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARALLAX)
parallax_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARALLAX)
parallax_scale = ti.field(dtype=ti.f32, shape=MAX_PARALLAX)

# Optional render-only dust particles. These are a faint atmospheric layer
# placed around the merger to suggest interstellar haze without altering
# the gravitational dynamics.
MAX_DUST = 1_200
dust_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_DUST)
dust_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_DUST)

# Optional render-only shockwave rings triggered near close nuclear passage.
# They are visual-only and do not change the dynamics.
MAX_SHOCK = 240
shock_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_SHOCK)
shock_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SHOCK)

# Optional render-only tidal streamers: faint, curved particle chains that
# echo the tidal tails during close passage without modifying the physics.
MAX_STREAMS = 320
stream_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_STREAMS)
stream_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_STREAMS)

# Optional render-only lens flares around nuclei for a brighter, more
# cinematic merger feel without affecting the dynamics.
MAX_FLARE = 160
flare_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_FLARE)
flare_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FLARE)

# Trail system: two pre-allocated 1D render fields for uploading one
# trail snapshot at a time.  At 12k stars each snapshot is ~96 KB --
# trivially small.  We keep a Python-side deque of numpy arrays and
# upload one per frame.  No per-frame GPU allocation needed.
TRAIL_FRAMES = 4             # echo frames kept in the deque
trail_render_p = ti.Vector.field(2, dtype=ti.f32, shape=MAX_N)  # upload target
trail_render_c = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)  # upload target

# ---------------------------------------------------------------------------
#  TAICHI UTILITY FUNCTIONS
# ---------------------------------------------------------------------------


@ti.func
def clamp01(x: ti.f32) -> ti.f32:
    return ti.max(0.0, ti.min(1.0, x))


@ti.func
def lerp3(a: ti.template(), b: ti.template(), t: ti.f32):
    return a * (1.0 - t) + b * t


@ti.func
def plummer_acc(dx: ti.f32, dy: ti.f32, mass: ti.f32) -> ti.Vector:
    """
    Compute gravitational acceleration from a point mass using
    Plummer softening to avoid singularity.

    a_vec = G * M / (r^2 + eps^2)^(3/2) * (delta_r_vec)

    dx, dy : displacement FROM particle TO attractor
    Returns acceleration vector pointing TOWARD the attractor.
    """
    r2 = dx * dx + dy * dy + EPS2          # (r^2 + eps^2)
    r3 = r2 * ti.sqrt(r2) + 1.0e-30        # (r^2 + eps^2)^(3/2)
    f = G_CONST * mass / r3
    return ti.Vector([f * dx, f * dy])

# ---------------------------------------------------------------------------
#  GALAXY INITIALISATION KERNEL
# ---------------------------------------------------------------------------


@ti.kernel
def init_galaxy_kernel(
    start:    ti.i32,        # first index in pos/vel/col
    count:    ti.i32,        # stars to place
    cx: ti.f32, cy: ti.f32,  # galaxy centre (world)
    vx: ti.f32, vy: ti.f32,  # bulk velocity
    tilt:     ti.f32,        # disk orientation (radians)
    nuc_idx:  ti.i32,        # which nucleus provides Keplerian potential
    # core colour
    cr: ti.f32, cg: ti.f32, cb: ti.f32,
    # outer disk colour
    dr: ti.f32, dg: ti.f32, db: ti.f32,
):
    col_core = ti.Vector([cr, cg, cb])
    col_disk = ti.Vector([dr, dg, db])

    for i in range(start, start + count):
        li = i - start   # 0..count-1

        # -- Radial profile: exponential surface density ---------------
        # Map uniform [0,1] sample u through the inverse CDF of an
        # exponential distribution: r ~ -h * ln(u)
        # This gives the correct n(r) ~ exp(-r/h) surface density.
        u = float(li) / float(count) + 1.0e-6
        r_raw = -R_DISK * 0.35 * ti.log(u)
        r_raw = ti.min(r_raw, R_DISK * 2.5)    # cap runaway wing

        # -- Logarithmic spiral arm angle ------------------------------
        arm = li % SPIRAL_ARMS
        phi0 = float(arm) / float(SPIRAL_ARMS) * 2.0 * math.pi
        phi = phi0 + WIND * (r_raw / R_DISK)

        # Scatter to give arms finite width (not knife edges)
        dr2 = (ti.random() - 0.5) * R_DISK * 0.25
        dphi = (ti.random() - 0.5) * 0.6
        r_fin = ti.abs(r_raw + dr2) + 1.0e-6
        phi += dphi

        # Vertical puffiness (projected as angular smear)
        dz = (ti.random() - 0.5) * 2.0 * DISK_THICK

        # Local disk coordinates
        lx = r_fin * ti.cos(phi)
        ly = r_fin * ti.sin(phi) + dz

        # Rotate disk by inclination angle
        rx = lx * ti.cos(tilt) - ly * ti.sin(tilt)
        ry = lx * ti.sin(tilt) + ly * ti.cos(tilt)

        # World position (small clamp keeps stars in [0,1]^2 initially)
        px = clamp01((cx + rx) * 0.95 + 0.025)
        py = clamp01((cy + ry) * 0.95 + 0.025)
        pos[i] = ti.Vector([px, py])

        # -- Circular Keplerian orbital velocity ----------------------
        # v_circ = sqrt(G * M_enc / r)
        # M_enc = nucleus mass + disk mass interior to r (linear approx)
        np2 = nuc_pos[nuc_idx]
        dp = ti.Vector([px - np2[0], py - np2[1]])
        r_orb = dp.norm() + EPS

        M_enc = nuc_mass[nuc_idx] + M_STAR * float(count) * (r_orb / R_DISK)
        v_circ = ti.sqrt(G_CONST * M_enc / (r_orb + EPS))

        # Tangent direction: perpendicular to radius, counter-clockwise
        tang = ti.Vector([-dp[1], dp[0]]) / (r_orb + EPS)
        vel[i] = ti.Vector([vx + tang[0] * v_circ,
                            vy + tang[1] * v_circ])

        # -- Per-vertex colour: core (hot) -> disk (young blue) -------
        t_col = clamp01(r_fin / R_DISK)
        c = lerp3(col_core, col_disk, t_col)

        # Young O/B stars in outer arms: boost blue channel
        blue_boost = clamp01((r_fin - 0.6 * R_DISK) / (0.4 * R_DISK)) * 0.28
        c[2] = clamp01(c[2] + blue_boost)

        # Surface-brightness peak at centre (like Sersic profile)
        brightness = clamp01(1.0 - 0.5 * t_col) + 0.18
        col[i] = c * brightness

        alive[i] = 1

# ---------------------------------------------------------------------------
#  PHYSICS STEP -- Semi-implicit Euler
# ---------------------------------------------------------------------------


@ti.kernel
def step_stars(
    dt:        ti.f32,
    n_stars:   ti.i32,
    n_nuc:     ti.i32,
    mx: ti.f32, my: ti.f32,    # mouse world-space position
    mouse_str: ti.f32,         # signed gravity strength (neg = repulsor)
):
    for i in range(n_stars):
        if alive[i] == 0:
            continue

        p = pos[i]
        v = vel[i]
        acc = ti.Vector([0.0, 0.0])

        # Sum Plummer-softened gravity from each active nucleus
        for k in range(n_nuc):
            if nuc_alive[k] == 0:
                continue
            np2 = nuc_pos[k]
            acc += plummer_acc(np2[0] - p[0], np2[1] - p[1], nuc_mass[k])

        # Mouse attractor / repulsor
        if mouse_str != 0.0:
            mdx = mx - p[0]
            mdy = my - p[1]
            mr2 = mdx * mdx + mdy * mdy
            if mr2 < MOUSE_R2:
                mr2s = mr2 + EPS2
                mr3 = mr2s * ti.sqrt(mr2s) + 1.0e-30
                mf = mouse_str / mr3
                acc += ti.Vector([mf * mdx, mf * mdy])

        # Semi-implicit Euler: velocity FIRST, then position
        v = v + acc * dt
        p = p + v * dt

        # Boundary respawn: keeps screen alive forever.
        # Stars that escape are re-seeded on a circular orbit around
        # the nearest nucleus, physically plausible as tidal debris.
        if p[0] < -0.15 or p[0] > 1.15 or p[1] < -0.15 or p[1] > 1.15:
            best = 0
            best2 = 1.0e18
            for k in range(n_nuc):
                if nuc_alive[k] == 0:
                    continue
                nd = nuc_pos[k] - p
                d2 = nd.dot(nd)
                if d2 < best2:
                    best2 = d2
                    best = k
            np2 = nuc_pos[best]
            ang = ti.random() * 2.0 * math.pi
            r_sp = R_DISK * (0.05 + ti.random() * 0.95)
            p = ti.Vector([np2[0] + r_sp * ti.cos(ang),
                           np2[1] + r_sp * ti.sin(ang)])
            vc = ti.sqrt(G_CONST * nuc_mass[best] / (r_sp + EPS))
            tang = ti.Vector([-ti.sin(ang), ti.cos(ang)])
            v = nuc_vel[best] + tang * vc

        pos[i] = p
        vel[i] = v

# ---------------------------------------------------------------------------
#  NUCLEUS STEP -- nuclei attract each other (merger dynamics)
# ---------------------------------------------------------------------------


@ti.kernel
def step_nuclei_kernel(dt: ti.f32, n_nuc: ti.i32):
    for k in range(n_nuc):
        if nuc_alive[k] == 0:
            continue
        p = nuc_pos[k]
        v = nuc_vel[k]
        acc = ti.Vector([0.0, 0.0])
        for j in range(n_nuc):
            if j == k or nuc_alive[j] == 0:
                continue
            np2 = nuc_pos[j]
            acc += plummer_acc(np2[0] - p[0], np2[1] - p[1], nuc_mass[j])
        v = v + acc * dt
        p = p + v * dt
        nuc_pos[k] = p
        nuc_vel[k] = v

# ---------------------------------------------------------------------------
#  TIDAL HEATING FLASH -- star colours whiten near pericentre passage
# ---------------------------------------------------------------------------


@ti.kernel
def apply_heat_flash(n: ti.i32, flash: ti.f32):
    """
    During close nuclear encounter, tidal shocks heat gas and stars,
    producing a starburst. We simulate this visually by blending all
    star colours toward hot white proportional to flash intensity.
    """
    white = ti.Vector([1.0, 0.92, 0.82])
    for i in range(n):
        if alive[i] == 0:
            continue
        col[i] = lerp3(col[i], white, flash * 0.10)

# ---------------------------------------------------------------------------
#  CINEMATIC COLOR GRADE -- optional artistic display transform
# ---------------------------------------------------------------------------


@ti.kernel
def apply_cinematic_grade(n: ti.i32, flash: ti.f32):
    """
    Optional render-only artistic color grade. This is a display transform,
    not physics: it does not modify pos, vel, or the persistent star colour
    field; it only prepares grade_col for drawing.
    """
    cool_shadow = ti.Vector([0.020, 0.045, 0.115])
    warm_high = ti.Vector([1.00, 0.78, 0.46])
    bloom_white = ti.Vector([1.00, 0.92, 0.78])
    for i in range(n):
        if alive[i] == 0:
            grade_col[i] = ti.Vector([0.0, 0.0, 0.0])
            continue

        c = col[i]
        luma = c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722

        # Artistic film grade: cool shadows, warm highlights, and a cheap
        # filmic shoulder so starburst flashes glow without flat clipping.
        shadow_t = clamp01((0.44 - luma) * 2.4)
        high_t = clamp01((luma - 0.44) * 1.9 + flash * 0.22)

        g = lerp3(c, cool_shadow, shadow_t * 0.24)
        g = lerp3(g, warm_high, high_t * 0.16)
        g = lerp3(g, bloom_white, flash * high_t * 0.10)

        gray = ti.Vector([luma, luma, luma])
        sat = 1.18 + flash * 0.10
        g = gray + (g - gray) * sat

        contrast = 1.14
        g = (g - ti.Vector([0.5, 0.5, 0.5])) * \
            contrast + ti.Vector([0.5, 0.5, 0.5])

        # Very cheap vignette, purely cinematic. It pulls attention toward
        # the merger while costing only a few scalar ops per visible star.
        p = pos[i] - ti.Vector([0.5, 0.5])
        r2 = p.dot(p)
        vignette = clamp01(1.06 - r2 * 1.05)
        g = g * vignette

        # Reinhard-style tone mapping keeps intense heated particles readable
        # on integrated GPUs without an extra post-process pass.
        exposure = 1.18 + flash * 0.10
        g = g * exposure
        g = g / (ti.Vector([1.0, 1.0, 1.0]) + g)
        g = g * 1.32

        grade_col[i] = ti.Vector([clamp01(g[0]), clamp01(g[1]), clamp01(g[2])])

# ---------------------------------------------------------------------------
#  ORBITAL-ENERGY COLOUR DIAGNOSTIC -- optional physical visualization
# ---------------------------------------------------------------------------


@ti.kernel
def apply_orbital_energy_colours(n: ti.i32, n_nuc: ti.i32, flash: ti.f32, use_grade: ti.i32):
    """
    Optional render-only diagnostic.

    Colour encodes local specific orbital energy relative to the nearest
    active nucleus:

        epsilon = 0.5 * |v - v_nucleus|^2 - G*M / sqrt(r^2 + eps^2)

    epsilon < 0 is locally bound; epsilon near 0 marks marginal tidal
    material; epsilon > 0 highlights escaping/high-energy debris. This is
    a physical diagnostic computed from existing state, not a force term.
    """
    bound_blue = ti.Vector([0.18, 0.48, 1.00])
    tidal_ice = ti.Vector([0.88, 0.96, 1.00])
    escape_hot = ti.Vector([1.00, 0.38, 0.12])

    for i in range(n):
        if alive[i] == 0:
            energy_col[i] = ti.Vector([0.0, 0.0, 0.0])
            continue

        p = pos[i]
        v = vel[i]

        best = 0
        best_d2 = 1.0e18
        for k in range(n_nuc):
            if nuc_alive[k] == 0:
                continue
            d = p - nuc_pos[k]
            d2 = d.dot(d)
            if d2 < best_d2:
                best_d2 = d2
                best = k

        rel_v = v - nuc_vel[best]
        speed2 = rel_v.dot(rel_v)
        r_soft = ti.sqrt(best_d2 + EPS2)
        epsilon = 0.5 * speed2 - G_CONST * nuc_mass[best] / (r_soft + 1.0e-30)

        bound_t = clamp01(-epsilon * 0.32)
        escape_t = clamp01(epsilon * 0.42)
        c = lerp3(tidal_ice, bound_blue, bound_t)
        c = lerp3(c, escape_hot, escape_t)

        # Brightness is tied to diagnostic strength, not invented physics:
        # marginal material stays pale; strongly bound/escaping stars stand out.
        signal = clamp01(ti.abs(epsilon) * 0.18)
        c = c * (0.58 + 0.42 * signal)

        if use_grade == 1:
            # Artistic display transform only; the physical quantity remains
            # the orbital-energy colour above.
            luma = c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
            gray = ti.Vector([luma, luma, luma])
            c = gray + (c - gray) * (1.12 + flash * 0.06)
            c = (c - ti.Vector([0.5, 0.5, 0.5])) * \
                1.08 + ti.Vector([0.5, 0.5, 0.5])
            centre = p - ti.Vector([0.5, 0.5])
            c = c * clamp01(1.04 - centre.dot(centre) * 0.85)
            c = c * (1.10 + flash * 0.08)
            c = c / (ti.Vector([1.0, 1.0, 1.0]) + c)
            c = c * 1.30

        energy_col[i] = ti.Vector(
            [clamp01(c[0]), clamp01(c[1]), clamp01(c[2])])

# ---------------------------------------------------------------------------
#  TWO-PASS STAR PARTICLES -- optional artistic render enhancement
# ---------------------------------------------------------------------------


@ti.kernel
def build_star_soft_colours(n: ti.i32, source_col: ti.template(), flash: ti.f32):
    """
    Optional render-only bloom pass for stars.

    Artistic layer: this creates a dim aura around each particle before the
    crisp core pass. It does not represent gas density, luminosity evolution,
    or any extra force; it only improves readability of sparse point sprites.
    """
    for i in range(n):
        if alive[i] == 0:
            star_soft_col[i] = ti.Vector([0.0, 0.0, 0.0])
            continue

        c = source_col[i]
        luma = c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
        halo = 0.18 + 0.10 * clamp01(luma * 1.8) + flash * 0.025
        star_soft_col[i] = ti.Vector([
            clamp01(c[0] * halo),
            clamp01(c[1] * halo),
            clamp01(c[2] * halo),
        ])


def draw_star_particles(canvas, colour_field, n_active: int, show_particle_bloom: bool, flash: float, use_camera: bool = False):
    """
    Draw live stars. Two-pass mode is optional and render-only:
    a faint larger pass creates subpixel presence, then a sharp core preserves
    the existing point-particle look.
    """
    target_pos = view_pos if use_camera else pos
    if show_particle_bloom:
        build_star_soft_colours(n_active, colour_field, flash)
        canvas.circles(target_pos, radius=0.0022,
                       per_vertex_color=star_soft_col)
        canvas.circles(target_pos, radius=0.00105,
                       per_vertex_color=colour_field)
    else:
        canvas.circles(target_pos, radius=0.0015,
                       per_vertex_color=colour_field)


@ti.kernel
def build_camera_view(n: ti.i32, n_nuc: ti.i32, cx: ti.f32, cy: ti.f32, zoom: ti.f32,
                      ox: ti.f32, oy: ti.f32):
    """
    Optional render-only camera transform. It pans, zooms, and gently orbits
    the visible particle positions around a dynamic focus point while leaving
    the physics state unchanged.
    """
    focus = ti.Vector([cx + ox, cy + oy])
    for i in range(n):
        if alive[i] == 0:
            continue
        p = pos[i]
        view_pos[i] = (p - focus) * zoom + ti.Vector([0.5, 0.5])

    for k in range(n_nuc):
        if nuc_alive[k] == 0:
            continue
        p = nuc_pos[k]
        view_nuc_pos[k] = (p - focus) * zoom + ti.Vector([0.5, 0.5])


def build_milky_way_background():
    """
    Procedural Milky Way background: a faint, curved band of softly glowing
    points placed behind the galaxies. It is artistic and render-only.
    """
    pts = np.zeros((MAX_MILKY, 2), dtype=np.float32)
    cols = np.zeros((MAX_MILKY, 3), dtype=np.float32)

    for i in range(MAX_MILKY):
        t = i / float(MAX_MILKY)
        band = 0.18 + 0.64 * t
        angle = math.pi * band

        x = 0.5 + 0.34 * math.cos(angle) + 0.025 * math.sin(8.0 * math.pi * t)
        y = 0.5 + 0.12 * math.sin(angle * 1.3) + \
            0.020 * math.cos(11.0 * math.pi * t)

        x += 0.06 * math.sin(3.0 * math.pi * t + 0.5)
        y += 0.04 * math.cos(2.5 * math.pi * t + 0.2)

        x = min(0.99, max(0.01, x))
        y = min(0.99, max(0.01, y))

        pts[i] = [x, y]
        brightness = 0.20 + 0.25 * (0.5 + 0.5 * math.sin(6.0 * math.pi * t))
        cols[i] = [0.16 + 0.18 * brightness,
                   0.20 + 0.16 * brightness,
                   0.28 + 0.14 * brightness]

    milky_pos.from_numpy(pts)
    milky_col.from_numpy(cols)


def build_parallax_stars():
    """
    Procedural parallax-star field. These are static, depth-sorted points with
    slight variation in colour and size to suggest a distant star field.
    """
    pts = np.zeros((MAX_PARALLAX, 2), dtype=np.float32)
    cols = np.zeros((MAX_PARALLAX, 3), dtype=np.float32)
    scales = np.zeros(MAX_PARALLAX, dtype=np.float32)
    rng = np.random.default_rng(7)

    for i in range(MAX_PARALLAX):
        angle = 2.0 * math.pi * rng.random()
        radius = 0.18 + 0.62 * rng.random()
        x = 0.5 + radius * math.cos(angle) + 0.05 * (rng.random() - 0.5)
        y = 0.5 + radius * 0.58 * math.sin(angle) + 0.04 * (rng.random() - 0.5)
        x = min(0.99, max(0.01, x))
        y = min(0.99, max(0.01, y))
        pts[i] = [x, y]

        brightness = 0.30 + 0.70 * rng.random()
        cols[i] = [0.55 + 0.25 * brightness,
                   0.60 + 0.22 * brightness,
                   0.85 + 0.12 * brightness]
        scales[i] = 0.00035 + 0.00045 * rng.random()

    parallax_pos.from_numpy(pts)
    parallax_col.from_numpy(cols)
    parallax_scale.from_numpy(scales)


def build_dust_particles():
    """
    Procedural dust cloud as an atmospheric overlay. It is artistic and does
    not contribute any force or density to the physics simulation.
    """
    pts = np.zeros((MAX_DUST, 2), dtype=np.float32)
    cols = np.zeros((MAX_DUST, 3), dtype=np.float32)
    rng = np.random.default_rng(11)

    for i in range(MAX_DUST):
        angle = 2.0 * math.pi * rng.random()
        radius = 0.10 + 0.32 * rng.random()
        x = 0.5 + radius * math.cos(angle) + 0.04 * (rng.random() - 0.5)
        y = 0.5 + radius * 0.55 * math.sin(angle) + 0.03 * (rng.random() - 0.5)
        x = min(0.99, max(0.01, x))
        y = min(0.99, max(0.01, y))
        pts[i] = [x, y]

        brightness = 0.14 + 0.18 * rng.random()
        cols[i] = [0.18 + 0.10 * brightness,
                   0.14 + 0.08 * brightness,
                   0.10 + 0.06 * brightness]

    dust_pos.from_numpy(pts)
    dust_col.from_numpy(cols)


def build_shockwave_rings(n_nuc: int, flash: float, frame: int):
    """
    Render-only shockwave rings around active nuclei during close passage.
    They are purely visual, inexpensive, and intended to echo the merger's
    intensity without altering the simulation state.
    """
    pts = np.zeros((MAX_SHOCK, 2), dtype=np.float32)
    cols = np.zeros((MAX_SHOCK, 3), dtype=np.float32)
    write_idx = 0
    strength = max(0.0, min(1.0, flash * 1.35))

    if strength <= 0.02:
        shock_pos.from_numpy(pts)
        shock_col.from_numpy(cols)
        return

    for k in range(n_nuc):
        if nuc_alive[k] == 0:
            continue
        p = nuc_pos[k]
        px = float(p[0])
        py = float(p[1])

        for ring in range(2):
            base_r = 0.016 + ring * 0.014 + strength * 0.028
            base_r += 0.0035 * math.sin(frame * 0.045 + ring * 0.7)
            for s in range(16):
                if write_idx >= MAX_SHOCK:
                    break
                theta = 2.0 * math.pi * (s / 16.0) + frame * 0.004 * (ring + 1)
                x = px + base_r * math.cos(theta)
                y = py + base_r * 0.72 * math.sin(theta)
                x = min(0.99, max(0.01, x))
                y = min(0.99, max(0.01, y))
                pts[write_idx] = [x, y]

                tint = 0.55 + 0.45 * strength
                if ring == 0:
                    cols[write_idx] = [0.95 * tint, 0.78 * tint, 0.42 * tint]
                else:
                    cols[write_idx] = [0.62 * tint, 0.85 * tint, 1.00 * tint]
                write_idx += 1

            if write_idx >= MAX_SHOCK:
                break

    shock_pos.from_numpy(pts)
    shock_col.from_numpy(cols)


def build_tidal_streamers(n_nuc: int, flash: float, frame: int):
    """
    Render-only curved streamer particles that echo the tidal tails of the
    merger. They are purely visual and inexpensive, so they stay optional.
    """
    pts = np.zeros((MAX_STREAMS, 2), dtype=np.float32)
    cols = np.zeros((MAX_STREAMS, 3), dtype=np.float32)
    write_idx = 0
    strength = max(0.0, min(1.0, flash * 1.25))

    if strength <= 0.02:
        stream_pos.from_numpy(pts)
        stream_col.from_numpy(cols)
        return

    for k in range(n_nuc):
        if nuc_alive[k] == 0:
            continue
        p = nuc_pos[k]
        px = float(p[0])
        py = float(p[1])

        for arm in range(2):
            phase = arm * math.pi + frame * 0.006 + k * 0.7
            for s in range(20):
                if write_idx >= MAX_STREAMS:
                    break
                t = s / 19.0
                radius = 0.015 + t * 0.055 + strength * 0.020
                theta = phase + (t - 0.5) * 1.25 + strength * 0.25
                x = px + radius * math.cos(theta)
                y = py + radius * 0.62 * math.sin(theta)
                x = min(0.99, max(0.01, x))
                y = min(0.99, max(0.01, y))
                pts[write_idx] = [x, y]

                tint = 0.55 + 0.45 * strength
                cols[write_idx] = [0.78 * tint, 0.60 * tint, 0.96 * tint]
                write_idx += 1

            if write_idx >= MAX_STREAMS:
                break

    stream_pos.from_numpy(pts)
    stream_col.from_numpy(cols)


def build_lens_flares(n_nuc: int, flash: float, frame: int):
    """
    Render-only lens flares around the nuclei. They are a cheap cinematic
    accent that responds to the merger flash without changing physics.
    """
    pts = np.zeros((MAX_FLARE, 2), dtype=np.float32)
    cols = np.zeros((MAX_FLARE, 3), dtype=np.float32)
    write_idx = 0
    strength = max(0.0, min(1.0, flash * 1.1))

    if strength <= 0.02:
        flare_pos.from_numpy(pts)
        flare_col.from_numpy(cols)
        return

    for k in range(n_nuc):
        if nuc_alive[k] == 0:
            continue
        p = nuc_pos[k]
        px = float(p[0])
        py = float(p[1])

        for s in range(8):
            if write_idx >= MAX_FLARE:
                break
            theta = 2.0 * math.pi * (s / 8.0) + frame * 0.004
            radius = 0.006 + strength * 0.018 + \
                0.002 * math.sin(frame * 0.06 + s)
            x = px + radius * math.cos(theta)
            y = py + radius * 0.58 * math.sin(theta)
            x = min(0.99, max(0.01, x))
            y = min(0.99, max(0.01, y))
            pts[write_idx] = [x, y]

            tint = 0.55 + 0.45 * strength
            cols[write_idx] = [1.00 * tint, 0.88 * tint, 0.72 * tint]
            write_idx += 1

        for s in range(4):
            if write_idx >= MAX_FLARE:
                break
            theta = 0.5 * math.pi * s + frame * 0.002
            radius = 0.004 + strength * 0.012
            x = px + radius * math.cos(theta)
            y = py + radius * 0.45 * math.sin(theta)
            x = min(0.99, max(0.01, x))
            y = min(0.99, max(0.01, y))
            pts[write_idx] = [x, y]
            cols[write_idx] = [0.95 * strength,
                               0.95 * strength, 1.00 * strength]
            write_idx += 1

    flare_pos.from_numpy(pts)
    flare_col.from_numpy(cols)


@ti.kernel
def build_nebula_points(n: ti.i32, n_nuc: ti.i32, flash: ti.f32) -> ti.i32:
    """
    Optional artistic nebula cloud built from the current stellar distribution.
    It is density-inspired because it samples the star field more densely in
    the inner, brighter regions and fades outward; it does not create new
    physics or alter gravitational forces.
    """
    write_idx = 0
    for i in range(n):
        if alive[i] == 0:
            continue

        # Sample every 8th star to keep the effect cheap on integrated GPUs.
        if i % 8 != 0:
            continue

        p = pos[i]
        c = col[i]

        best = 0
        best_d2 = 1.0e18
        for k in range(n_nuc):
            if nuc_alive[k] == 0:
                continue
            d = p - nuc_pos[k]
            d2 = d.dot(d)
            if d2 < best_d2:
                best_d2 = d2
                best = k

        # Keep the nebula concentrated near the galactic nuclei.
        if best_d2 > 0.15 * 0.15:
            continue

        # A simple density-like weight from proximity to the core.
        weight = 1.0 - ti.sqrt(best_d2) / 0.15
        weight = clamp01(weight)
        if weight < 0.08:
            continue

        if write_idx < MAX_NEBULA:
            jitter = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.018
            nebula_pos[write_idx] = p + jitter * (0.45 + weight * 0.85)
            nebula_col[write_idx] = (
                c * (0.16 + 0.28 * weight) + nuc_col[best] * (0.08 + 0.16 * weight)) * (1.0 + flash * 0.12)
            write_idx += 1

    return write_idx

# ---------------------------------------------------------------------------
#  FIELD-LINE OVERLAY (static, computed once)
# ---------------------------------------------------------------------------


def build_field_lines():
    """
    Draw faint logarithmic spiral field lines tracing the potential
    geometry of each galaxy at t=0. Updated from Python then uploaded.
    """
    configs = [
        (GAL1_PX, GAL1_PY, GAL1_TILT, 1.0, 0.55, 0.08),
        (GAL2_PX, GAL2_PY, GAL2_TILT, 0.05, 0.90, 0.80),
    ]
    per = N_FIELD // len(configs)
    pts = np.zeros((N_FIELD, 2), dtype=np.float32)
    cols = np.zeros((N_FIELD, 3), dtype=np.float32)

    for gid, (cx, cy, tilt, cr, cg, cb) in enumerate(configs):
        for j in range(per):
            t = (j + 1) / per
            r_f = R_DISK * 1.1 * math.sqrt(t)
            arm = j % SPIRAL_ARMS
            phi = (arm / SPIRAL_ARMS) * 2 * math.pi + WIND * (r_f / R_DISK)
            lx = r_f * math.cos(phi)
            ly = r_f * math.sin(phi)
            rx = lx * math.cos(tilt) - ly * math.sin(tilt)
            ry = lx * math.sin(tilt) + ly * math.cos(tilt)
            idx = gid * per + j
            pts[idx] = [cx + rx, cy + ry]
            br = 0.04 + 0.05 * (1.0 - t)
            cols[idx] = [cr * br, cg * br, cb * br]

    fl_pos.from_numpy(pts)
    fl_col.from_numpy(cols)

# ---------------------------------------------------------------------------
#  GLOW HALO BUILDER (called each frame -- tracks moving nuclei)
# ---------------------------------------------------------------------------


# Slim glow for i3: 4 layers, 32 angular steps instead of 7 x 64
RING_LAYERS = [
    (0.34, 1.00), (0.72, 0.48), (1.28, 0.20), (2.05, 0.075),
]
GLOW_STEPS = 32

# Precompute ring directions once. The glow effect is artistic, but it is
# driven by real nucleus position, velocity, colour, and merger flash.
GLOW_THETA = np.linspace(0.0, 2.0 * math.pi, GLOW_STEPS,
                         endpoint=False, dtype=np.float32)
GLOW_UNIT = np.stack((np.cos(GLOW_THETA), np.sin(
    GLOW_THETA)), axis=1).astype(np.float32)
GLOW_SHIMMER = (0.78 + 0.22 * (1.0 + np.cos(GLOW_THETA * 2.0))
                * 0.5).astype(np.float32)
GLOW_POS_NP = np.empty((MAX_GLOW, 2), dtype=np.float32)
GLOW_COL_NP = np.zeros((MAX_GLOW, 3), dtype=np.float32)


def build_glow(n_nuc: int, flash: float) -> int:
    """
    Approximate a glowing nuclear halo by placing many dim points on
    concentric rings. Returns number of glow points written.

    Artistic layer: the halo is elongated along nucleus motion to suggest
    lens bloom and hot accretion light. It is render-only and does not feed
    back into the gravitational simulation.
    """
    GLOW_POS_NP.fill(-1.0)
    GLOW_COL_NP.fill(0.0)
    write_idx = 0

    for k in range(n_nuc):
        if int(nuc_alive[k]) == 0:
            continue
        p = nuc_pos[k]
        v = nuc_vel[k]
        c = nuc_col[k]
        cx = float(p[0])
        cy = float(p[1])
        vx = float(v[0])
        vy = float(v[1])
        cr = float(c[0])
        cg = float(c[1])
        cb = float(c[2])
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > 1.0e-7:
            ax = vx / speed
            ay = vy / speed
        else:
            ax = 1.0
            ay = 0.0

        # Mild motion-aligned stretch: enough to feel cinematic, still cheap.
        stretch = min(0.55, speed * 28.0 + flash * 0.18)
        radius = 0.023 * (1.0 + 0.26 * flash)
        mass_boost = min(1.0, float(nuc_mass[k]) / M_NUCLEUS)

        for (scale, alpha) in RING_LAYERS:
            if write_idx + GLOW_STEPS > MAX_GLOW:
                break
            a_mod = alpha * (0.70 + 0.30 * mass_boost) * (1.0 + flash * 0.55)
            for s in range(GLOW_STEPS):
                ux = float(GLOW_UNIT[s, 0])
                uy = float(GLOW_UNIT[s, 1])
                along = ux
                perp = uy
                px = ax * along * (1.0 + stretch) - ay * \
                    perp * (1.0 - stretch * 0.28)
                py = ay * along * (1.0 + stretch) + ax * \
                    perp * (1.0 - stretch * 0.28)
                ray = max(0.0, along)
                shimmer = float(GLOW_SHIMMER[s])
                idx = write_idx + s
                GLOW_POS_NP[idx] = [cx + radius * scale * px,
                                    cy + radius * scale * py]
                GLOW_COL_NP[idx] = [cr * a_mod * shimmer * (1.0 + 0.10 * ray),
                                    cg * a_mod * shimmer * (1.0 + 0.06 * ray),
                                    cb * a_mod * shimmer]
            write_idx += GLOW_STEPS

    if write_idx == 0:
        return 0

    glow_pos.from_numpy(GLOW_POS_NP)
    glow_col.from_numpy(GLOW_COL_NP)
    return write_idx


def draw_nucleus_cores(canvas, flash: float, n_nuc: int, use_camera: bool = False):
    """
    Render-only nucleus polish: a soft outer corona plus a tighter inner
    core, both driven by the real nucleus state. This is purely artistic and
    does not affect the gravitational dynamics.
    """
    for idx in range(3):
        nuc_dot_pos[idx] = [2.0, 2.0]
        nuc_dot_col[idx] = [0.0, 0.0, 0.0]

    n_dots = 0
    for k in range(n_nuc):
        if not nuc_alive[k]:
            continue
        p = view_nuc_pos[k] if use_camera else nuc_pos[k]
        c = nuc_col[k]
        pulse = 1.0 + flash * 0.65
        nuc_dot_pos[n_dots] = [float(p[0]), float(p[1])]
        nuc_dot_col[n_dots] = [
            min(1.0, float(c[0]) * pulse),
            min(1.0, float(c[1]) * pulse),
            min(1.0, float(c[2]) * pulse),
        ]
        n_dots += 1

    if n_dots > 0:
        # Outer corona: soft, large, and slightly desaturated.
        canvas.circles(nuc_dot_pos, radius=0.0098,
                       per_vertex_color=nuc_dot_col)
        # Inner shell: brighter and more focused.
        canvas.circles(nuc_dot_pos, radius=0.0050,
                       per_vertex_color=nuc_dot_col)
        # Hot point core: crisp and visible on integrated GPUs.
        canvas.circles(nuc_dot_pos, radius=0.0022,
                       per_vertex_color=nuc_dot_col)

# ---------------------------------------------------------------------------
#  RESET HELPER
# ---------------------------------------------------------------------------


def full_reset():
    # --- Nuclei ---
    nuc_pos[0] = [GAL1_PX, GAL1_PY]
    nuc_vel[0] = [GAL1_VX, GAL1_VY]
    nuc_mass[0] = M_NUCLEUS
    nuc_col[0] = [1.0, 0.72, 0.18]    # warm gold
    nuc_alive[0] = 1

    nuc_pos[1] = [GAL2_PX, GAL2_PY]
    nuc_vel[1] = [GAL2_VX, GAL2_VY]
    nuc_mass[1] = M_NUCLEUS
    nuc_col[1] = [0.10, 0.90, 1.00]   # hot cyan
    nuc_alive[1] = 1

    nuc_pos[2] = [0.5, 0.5]
    nuc_vel[2] = [0.0, 0.0]
    nuc_mass[2] = 0.0
    nuc_col[2] = [0.80, 0.50, 1.00]
    nuc_alive[2] = 0

    # --- Galaxy 1: gold core -> icy blue spiral arms ---
    init_galaxy_kernel(
        0, N_PER_GAL,
        GAL1_PX, GAL1_PY, GAL1_VX, GAL1_VY,
        GAL1_TILT, 0,
        1.00, 0.65, 0.10,    # core: molten gold
        0.30, 0.60, 1.00,    # disk: icy steel blue
    )

    # --- Galaxy 2: cyan core -> deep violet spiral arms ---
    init_galaxy_kernel(
        N_PER_GAL, N_PER_GAL,
        GAL2_PX, GAL2_PY, GAL2_VX, GAL2_VY,
        GAL2_TILT, 1,
        0.05, 0.95, 0.85,    # core: hot cyan
        0.70, 0.25, 1.00,    # disk: violet
    )

    # Deactivate dwarf-galaxy star slots
    for i in range(N_TOTAL, MAX_N):
        alive[i] = 0

# ---------------------------------------------------------------------------
#  MAIN LOOP
# ---------------------------------------------------------------------------


def main():
    print(__doc__)
    print("=" * 60)
    print(f"Stars: {N_TOTAL:,}  |  Window: {WINDOW_W}x{WINDOW_H}")
    print("Initialising...")

    full_reset()
    build_field_lines()
    build_milky_way_background()
    build_parallax_stars()
    build_dust_particles()

    window = ti.ui.Window(
        "Galaxy Collision -- Real Gravitational N-Body Physics",
        (WINDOW_W, WINDOW_H),
        vsync=True,
    )
    canvas = window.get_canvas()

    # Simulation state
    paused = False
    show_trails = False       # OFF by default -- expensive on i3
    show_glow = True
    show_fields = False
    show_nebula = False      # N key: optional artistic nebula cloud
    show_camera = False      # M key: optional automatic cinematic camera
    show_milky = False       # H key: optional procedural Milky Way background
    show_parallax = False    # J key: optional parallax stars
    show_dust = False        # K key: optional dust particles
    show_shockwaves = False  # Y key: optional shockwave rings
    show_streamers = False   # U key: optional tidal streamers
    show_lens_flares = False  # I key: optional lens flares
    show_grade = True        # render-only cinematic grade, C toggles
    show_energy = False       # E key: physical orbital-energy diagnostic
    show_particle_bloom = True  # B key: optional two-pass star rendering
    perf_mode = False       # P key: ultra-minimal rendering

    # Phase 5 (optional): Adaptive Quality Controller (render-only)
    # Toggle: O
    # This is purely artistic/render quality adaptation to keep FPS stable.
    adaptive_quality = False
    adaptive_key_prev = False  # internal debounce
    aq_target_fps = 30.0
    aq_low_fps = 26.0
    aq_high_fps = 33.0
    aq_last_adjust_frame = -999999
    aq_user_restore = {
        'show_trails': False,
        'show_glow': False,
        'show_nebula': False,
        'show_shockwaves': False,
        'show_streamers': False,
        'show_lens_flares': False,
        'show_particle_bloom': True,
    }
    aq_restore_valid = False

    dt_scale = 1.0

    n_nuc = 2
    n_active = N_TOTAL
    dwarf_spawned = False
    flash = 0.0
    frame = 0
    cam_center_x = 0.5
    cam_center_y = 0.5
    cam_zoom = 1.0
    cam_orbit_angle = 0.0
    # Trail deque: Python-side ring buffer of numpy snapshots.
    # Each entry is (pos_np [N,2], col_np [N,3]) float32.
    # At 12k stars: 12000 * (2+3) * 4 bytes = ~240 KB per frame -- tiny.
    # We upload one entry per render cycle to trail_render_p/c (pre-alloc 1D fields).
    import collections
    trail_deque = collections.deque(maxlen=TRAIL_FRAMES)

    fps_buf = []
    t_last = time.time()

    print("Window open. Enjoy the collision!")
    print("=" * 60)

    while window.running:
        # -- FPS tracking ---------------------------------------------
        t_now = time.time()
        elapsed = t_now - t_last + 1.0e-9
        t_last = t_now
        fps_buf.append(1.0 / elapsed)
        if len(fps_buf) > 60:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        # -- Input ----------------------------------------------------
        mx, my = window.get_cursor_pos()
        mouse_str = 0.0

        if window.get_event(ti.ui.PRESS):
            k = window.event.key
            if k == ti.ui.SPACE:
                paused = not paused
                print("Paused." if paused else "Resumed.")
            elif k == 't':
                show_trails = not show_trails
                print("Trails:", "ON (may lower FPS)" if show_trails else "OFF")
            elif k == 'f':
                show_fields = not show_fields
                print("Field lines:", "ON" if show_fields else "OFF")
            elif k == 'g':
                show_glow = not show_glow
                print("Glow:", "ON" if show_glow else "OFF")
            elif k == 'n':
                show_nebula = not show_nebula
                print("Nebula:", "ON" if show_nebula else "OFF")
            elif k == 'm':
                show_camera = not show_camera
                print("Cinematic camera:", "ON" if show_camera else "OFF")
            elif k in ('h', 'H'):
                show_milky = not show_milky
                print("Milky Way background:", "ON" if show_milky else "OFF")
            elif k in ('j', 'J'):
                show_parallax = not show_parallax
                print("Parallax stars:", "ON" if show_parallax else "OFF")
            elif k in ('k', 'K'):
                show_dust = not show_dust
                print("Dust particles:", "ON" if show_dust else "OFF")
            elif k in ('y', 'Y'):
                show_shockwaves = not show_shockwaves
                print("Shockwave rings:", "ON" if show_shockwaves else "OFF")
            elif k in ('u', 'U'):
                show_streamers = not show_streamers
                print("Tidal streamers:", "ON" if show_streamers else "OFF")
            elif k in ('i', 'I'):
                show_lens_flares = not show_lens_flares
                print("Lens flares:", "ON" if show_lens_flares else "OFF")
            elif k == 'c':
                show_grade = not show_grade
                print("Cinematic grade:", "ON" if show_grade else "OFF")
            elif k == 'e':
                show_energy = not show_energy
                print("Orbital-energy colours:",
                      "ON" if show_energy else "OFF")
            elif k == 'b':
                show_particle_bloom = not show_particle_bloom
                print("Particle bloom:", "ON" if show_particle_bloom else "OFF")
            elif k == 'o' or k == 'O':
                # Phase 5: Adaptive quality controller (render-only)
                adaptive_quality = not adaptive_quality
                # When turning ON, restore any user-chosen FX states.
                aq_restore_valid = False
                print("Adaptive quality:", "ON" if adaptive_quality else "OFF")
            elif k == 'p':

                perf_mode = not perf_mode
                if perf_mode:
                    show_trails = False
                    show_glow = False
                    show_fields = False
                    show_nebula = False
                    show_milky = False
                    show_parallax = False
                    show_dust = False
                    show_shockwaves = False
                    show_streamers = False
                    show_lens_flares = False
                    show_grade = False
                    show_energy = False
                    show_particle_bloom = False
                    print("Performance mode ON -- trails/glow disabled for max FPS")
                else:
                    show_glow = True
                    show_grade = True
                    show_particle_bloom = True
                    print("Performance mode OFF")
            elif k in ('=', '+'):
                dt_scale = min(dt_scale * 1.25, 8.0)
                print(f"Time scale: {dt_scale:.2f}x")
            elif k == '-':
                dt_scale = max(dt_scale * 0.80, 0.1)
                print(f"Time scale: {dt_scale:.2f}x")
            elif k == 'r':
                full_reset()
                n_active = N_TOTAL
                n_nuc = 2
                dwarf_spawned = False
                trail_deque.clear()
                flash = 0.0
                frame = 0
                print("Reset complete.")
            elif k == '1' and not dwarf_spawned:
                nuc_pos[2] = [mx, my]
                nuc_vel[2] = [GAL1_VX * 0.35, -GAL1_VY * 0.55]
                nuc_mass[2] = M_NUCLEUS * 0.30
                nuc_col[2] = [0.80, 0.50, 1.00]
                nuc_alive[2] = 1
                init_galaxy_kernel(
                    N_TOTAL, N_PER_GAL // 3,
                    mx, my, 0.0, 0.0, 1.2, 2,
                    0.80, 0.50, 1.00,
                    0.40, 0.18, 0.90,
                )
                n_active = N_TOTAL + N_PER_GAL // 3
                n_nuc = 3
                dwarf_spawned = True
                print(f"Dwarf galaxy spawned at ({mx:.2f}, {my:.2f})!")
            elif k == 'q':
                break

        if window.is_pressed(ti.ui.LMB):
            mouse_str = MOUSE_G * M_NUCLEUS * 15.0
        elif window.is_pressed(ti.ui.RMB):
            mouse_str = REPULSOR_G * M_NUCLEUS * 12.0

        # -- Physics --------------------------------------------------
        if not paused:

            dt = DT_BASE * dt_scale
            substeps = max(1, int(dt_scale * 1.5))
            sub_dt = dt / substeps

            for _ in range(substeps):
                step_nuclei_kernel(sub_dt, n_nuc)
                step_stars(sub_dt, n_active, n_nuc, mx, my, mouse_str)

            # Tidal heating: compute flash from nuclear separation
            if n_nuc >= 2 and nuc_alive[0] and nuc_alive[1]:
                d = nuc_pos[0] - nuc_pos[1]
                dist = math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2)
                tgt = max(0.0, 1.0 - dist / (R_DISK * 1.5))
                flash = flash * 0.88 + tgt * 0.12
                if flash > 0.015:
                    apply_heat_flash(n_active, flash)

            # Trail capture every 4 frames: snapshot pos/col to numpy deque.
            # to_numpy() on 12k-star fields is ~96 KB -- fast and safe.
            if show_trails and frame % 4 == 0:
                trail_deque.append((
                    pos.to_numpy()[:n_active].copy(),
                    col.to_numpy()[:n_active].copy(),
                ))

            frame += 1

        # -- Adaptive Quality Controller (render-only) --------------
        # Runs even while paused so the view stays responsive.
        if adaptive_quality:
            # Adjust at a low rate to avoid flicker.
            if (frame - aq_last_adjust_frame) > 10 and len(fps_buf) >= 30:
                # If FPS is too low, disable the heaviest render layers.
                if fps < aq_low_fps:
                    if not aq_restore_valid:
                        aq_user_restore = {
                            'show_trails': show_trails,
                            'show_glow': show_glow,
                            'show_nebula': show_nebula,
                            'show_shockwaves': show_shockwaves,
                            'show_streamers': show_streamers,
                            'show_lens_flares': show_lens_flares,
                            'show_particle_bloom': show_particle_bloom,
                        }
                        aq_restore_valid = True

                    # Drop expensive render workload first.
                    show_particle_bloom = False
                    show_shockwaves = False
                    show_streamers = False
                    show_lens_flares = False

                    # If still struggling, also drop glow/trails.
                    if fps < (aq_low_fps - 2.0):
                        show_glow = False
                        show_trails = False

                    aq_last_adjust_frame = frame

                # If FPS recovered, restore user intent gradually.
                elif fps > aq_high_fps and aq_restore_valid:
                    show_particle_bloom = aq_user_restore['show_particle_bloom']
                    show_shockwaves = aq_user_restore['show_shockwaves']
                    show_streamers = aq_user_restore['show_streamers']
                    show_lens_flares = aq_user_restore['show_lens_flares']

                    show_glow = aq_user_restore['show_glow']
                    show_trails = aq_user_restore['show_trails']
                    show_nebula = aq_user_restore['show_nebula']

                    aq_restore_valid = False
                    aq_last_adjust_frame = frame

        # -- Render ---------------------------------------------------
        canvas.set_background_color((0.007, 0.007, 0.016))

        if show_camera:
            if n_nuc >= 2 and nuc_alive[0] and nuc_alive[1]:
                d = nuc_pos[0] - nuc_pos[1]
                sep = math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2)
                focus_x = 0.5 * (float(nuc_pos[0][0]) + float(nuc_pos[1][0]))
                focus_y = 0.5 * (float(nuc_pos[0][1]) + float(nuc_pos[1][1]))
                zoom_target = 1.0 + 0.45 * max(0.0, 1.0 - min(1.0, sep / 0.16))
            else:
                focus_x = 0.5
                focus_y = 0.5
                zoom_target = 1.0

            cam_center_x = cam_center_x * 0.86 + focus_x * 0.14
            cam_center_y = cam_center_y * 0.86 + focus_y * 0.14
            cam_zoom = cam_zoom * 0.84 + zoom_target * 0.16
            cam_orbit_angle += 0.018 * (1.0 + 0.25 * flash)
            orbit_strength = 0.035 * (1.0 + 0.08 * cam_zoom)
            orbit_x = math.sin(cam_orbit_angle) * orbit_strength
            orbit_y = math.cos(cam_orbit_angle * 0.7) * orbit_strength * 0.55
            build_camera_view(n_active, n_nuc, cam_center_x,
                              cam_center_y, cam_zoom, orbit_x, orbit_y)

        # Field-line overlay (static)
        if show_fields:
            canvas.circles(fl_pos, radius=0.0014, per_vertex_color=fl_col)

        # Trail render: upload each deque snapshot to pre-allocated 1D fields.
        # Oldest snapshot drawn first (darkest), newest last (brightest).
        # canvas.circles() on a 1D field is perfectly supported.
        if show_trails and len(trail_deque) > 0:
            n_snaps = len(trail_deque)
            for snap_idx, (pnp, cnp) in enumerate(trail_deque):
                t_frac = (snap_idx + 1) / n_snaps       # 0=oldest, 1=newest
                alpha = t_frac * 0.28
                trad = 0.0007 + 0.0006 * t_frac
                trail_render_p.from_numpy(pnp)
                trail_render_c.from_numpy((cnp * alpha).astype(np.float32))
                canvas.circles(trail_render_p, radius=trad,
                               per_vertex_color=trail_render_c)

        # Optional artistic nebula cloud
        if show_nebula:
            n_neb = build_nebula_points(n_active, n_nuc, flash)
            if n_neb > 0:
                canvas.circles(nebula_pos, radius=0.0080,
                               per_vertex_color=nebula_col)
                canvas.circles(nebula_pos, radius=0.0042,
                               per_vertex_color=nebula_col)

        # Optional procedural Milky Way backdrop
        if show_milky:
            canvas.circles(milky_pos, radius=0.0018,
                           per_vertex_color=milky_col)

        # Optional parallax stars
        if show_parallax:
            canvas.circles(parallax_pos, radius=0.0010,
                           per_vertex_color=parallax_col)
            canvas.circles(parallax_pos, radius=0.00045,
                           per_vertex_color=parallax_col)

        # Optional dust particles
        if show_dust:
            canvas.circles(dust_pos, radius=0.00075,
                           per_vertex_color=dust_col)

        # Optional shockwave rings around nuclei
        if show_shockwaves:
            build_shockwave_rings(n_nuc, flash, frame)
            canvas.circles(shock_pos, radius=0.0022,
                           per_vertex_color=shock_col)

        # Optional tidal streamers around nuclei
        if show_streamers:
            build_tidal_streamers(n_nuc, flash, frame)
            canvas.circles(stream_pos, radius=0.00085,
                           per_vertex_color=stream_col)

        # Optional lens flares around nuclei
        if show_lens_flares:
            build_lens_flares(n_nuc, flash, frame)
            canvas.circles(flare_pos, radius=0.0016,
                           per_vertex_color=flare_col)

        # Live stars
        if show_energy:
            apply_orbital_energy_colours(
                n_active, n_nuc, flash, 1 if show_grade else 0)
            draw_star_particles(canvas, energy_col, n_active,
                                show_particle_bloom, flash,
                                use_camera=show_camera)
        elif show_grade:
            apply_cinematic_grade(n_active, flash)
            draw_star_particles(canvas, grade_col, n_active,
                                show_particle_bloom, flash,
                                use_camera=show_camera)
        else:
            draw_star_particles(canvas, col, n_active,
                                show_particle_bloom, flash,
                                use_camera=show_camera)

        # Glow halos + bright nucleus cores
        if show_glow:
            n_gp = build_glow(n_nuc, flash)
            if n_gp > 0:
                canvas.circles(glow_pos, radius=0.0030,
                               per_vertex_color=glow_col)

            draw_nucleus_cores(canvas, flash, n_nuc, use_camera=show_camera)

        # Frame title with telemetry
        sep_str = ""
        if n_nuc >= 2 and nuc_alive[0] and nuc_alive[1]:
            d = nuc_pos[0] - nuc_pos[1]
            sep = math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2)
            sep_str = f"  |  Core sep: {sep:.3f}"

        window.show()

    print("\nSimulation ended. Thank you for watching the universe collide.")


if __name__ == "__main__":
    main()
