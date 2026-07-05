# -*- coding: utf-8 -*-
"""
THREE-BODY PROBLEM - Cinematic Physics Simulation
Real Newtonian gravity, Chaos, Leapfrog integration

CONTROLS
  Mouse Left-click/drag  - Apply gravitational pull toward cursor
  Mouse Right-click      - Spawn a burst of test particles at cursor
  SPACE                  - Pause / Resume
  T                      - Toggle trails on/off
  R                      - Reset to initial conditions
  F                      - Cycle through preset configurations
  +  /  =                - Speed up time
  -                      - Slow down time
  1  /  2  /  3          - Highlight body 1/2/3 (Hill sphere ring)
  ESC                    - Quit
"""

import taichi as ti
import math, time

# --- Taichi initialisation ---
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32)
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    print("Running on CPU - lower FPS expected. Reduce N_TEST if it crawls.")

print(__doc__)

# --- Simulation constants ---
WIDTH,  HEIGHT   = 1280, 800
SCALE            = 300.0         # world-units to pixels
CX, CY           = WIDTH//2, HEIGHT//2

G                = 1.0           # gravitational constant (normalised)
BODY_MASS        = 1.0           # mass of each massive body
SOFTENING_SQ     = (0.02)**2     # epsilon^2 - prevents 1/r singularity

N_TEST           = 12000         # number of massless test particles
TRAIL_LEN        = 320           # ring-buffer length for body trails

DT_BASE          = 2e-3          # base time-step per frame
SUBSTEPS         = 6             # leapfrog sub-steps per rendered frame
TIME_SCALE_INIT  = 1.0           # initial time multiplier

BODY_COLORS = [
    [1.00, 0.45, 0.10],   # body 0: warm amber
    [0.20, 0.70, 1.00],   # body 1: electric blue
    [0.85, 0.20, 0.90],   # body 2: violet/magenta
]

# --- Famous preset configurations ---
PRESETS = {
    # Figure-8 choreography (Chenciner & Montgomery 2000) - perfectly periodic
    "figure8": [
        [ 0.9700436, -0.2430872,  0.46620368,  0.43236573],
        [-0.9700436,  0.2430872,  0.46620368,  0.43236573],
        [ 0.0,        0.0,       -0.93240737, -0.86473146],
    ],
    # Lagrange equilateral triangle - stable in equal-mass case
    "lagrange": [
        [ 1.0,  0.0,       0.0,  0.5773503],
        [-0.5,  0.8660254, 0.5, -0.2886751],
        [-0.5, -0.8660254,-0.5, -0.2886751],
    ],
    # Broucke-Henon: two bodies oscillate, third traces outer oval
    "broucke": [
        [-0.9892620043,  0.0,  0.0,  1.9169244185],
        [ 2.2096177241,  0.0,  0.0,  0.1910268739],
        [-1.2203557197,  0.0,  0.0, -2.1079512924],
    ],
    # Chaotic - sensitive dependence on initial conditions
    "chaos": [
        [ 0.97,  0.24,  0.93,  0.86],
        [-0.97, -0.24,  0.93,  0.86],
        [ 0.0,   0.0,  -1.86, -1.72],
    ],
    # Yin-Yang I (Sun & Liao 2013) - discovered via deep continuation
    "yinyang": [
        [ 0.513938,  0.304736, -0.974090,  0.783260],
        [-0.513938, -0.304736, -0.974090,  0.783260],
        [ 0.0,       0.0,      1.948180, -1.566520],
    ],
}
PRESET_NAMES = list(PRESETS.keys())

# --- Taichi fields ---
bpos  = ti.Vector.field(2, dtype=ti.f32, shape=3)   # body positions
bvel  = ti.Vector.field(2, dtype=ti.f32, shape=3)   # body velocities
bacc  = ti.Vector.field(2, dtype=ti.f32, shape=3)   # body accelerations
bmass = ti.field(dtype=ti.f32, shape=3)              # body masses

tpos   = ti.Vector.field(2, dtype=ti.f32, shape=N_TEST)  # particle positions
tvel   = ti.Vector.field(2, dtype=ti.f32, shape=N_TEST)  # particle velocities
tcol   = ti.Vector.field(3, dtype=ti.f32, shape=N_TEST)  # particle RGB colours
talive = ti.field(dtype=ti.i32, shape=N_TEST)             # particle alive flag

trail_pos  = ti.Vector.field(2, dtype=ti.f32, shape=(3, TRAIL_LEN))
trail_head = ti.field(dtype=ti.i32, shape=3)   # ring-buffer write head

pixels = ti.Vector.field(4, dtype=ti.f32, shape=(WIDTH, HEIGHT))  # HDR RGBA

bcolor = ti.Vector.field(3, dtype=ti.f32, shape=3)   # body RGB colours (Taichi field for kernel access)

mouse_world = ti.Vector.field(2, dtype=ti.f32, shape=())
mouse_pull  = ti.field(dtype=ti.f32, shape=())   # 1 = pull active
paused      = ti.field(dtype=ti.i32, shape=())

# Initialise body colour field from Python list
for _bi in range(3):
    bcolor[_bi] = BODY_COLORS[_bi]

# --- Physics kernels ---

@ti.func
def grav_acc_fn(p, q, m):
    # Softened Newtonian gravity: a = G*m*(q-p) / (|q-p|^2 + eps^2)^(3/2)
    r  = q - p
    d2 = r.dot(r) + SOFTENING_SQ
    d3 = ti.sqrt(d2) * d2        # |r|^3 with softening
    return G * m / d3 * r


@ti.kernel
def integrate_bodies(dt: float):
    # Kick-Drift-Kick leapfrog for the 3 massive bodies
    # Kick: v += a*dt/2
    for i in range(3):
        bvel[i] += bacc[i] * (dt * 0.5)
    # Drift: x += v*dt
    for i in range(3):
        bpos[i] += bvel[i] * dt
    # Recompute accelerations from new positions
    for i in range(3):
        a = ti.Vector([0.0, 0.0])
        for j in range(3):
            if i != j:
                a += grav_acc_fn(bpos[i], bpos[j], bmass[j])
        bacc[i] = a
    # Kick again: v += a*dt/2
    for i in range(3):
        bvel[i] += bacc[i] * (dt * 0.5)


@ti.kernel
def integrate_particles(dt: float):
    # Symplectic Euler for massless test particles under body gravity
    for i in range(N_TEST):
        if talive[i] == 0:
            continue
        p   = tpos[i]
        acc = ti.Vector([0.0, 0.0])
        for j in range(3):
            acc += grav_acc_fn(p, bpos[j], bmass[j])
        # Optional mouse pull (virtual attractor mass = 2.5)
        if mouse_pull[None] > 0.5:
            acc += grav_acc_fn(p, mouse_world[None], 2.5)
        tvel[i] += acc * dt
        tpos[i] += tvel[i] * dt
        d2 = tpos[i].dot(tpos[i])
        if d2 > 25.0:   # particle escaped beyond radius 5 world units
            talive[i] = 0


@ti.kernel
def respawn_particles(seed: int):
    # Respawn dead particles near a random body with quasi-circular velocity
    for i in range(N_TEST):
        if talive[i] == 0:
            bi  = (i * 7 + seed) % 3
            r   = 0.08 + ti.random(ti.f32) * 0.25
            ang = ti.random(ti.f32) * 6.2831853
            tpos[i] = bpos[bi] + ti.Vector([r * ti.cos(ang), r * ti.sin(ang)])
            speed   = ti.sqrt(G * bmass[bi] / (r + 0.02))
            tang    = ti.Vector([-ti.sin(ang), ti.cos(ang)])
            tvel[i] = bvel[bi] + tang * speed * (0.7 + ti.random(ti.f32) * 0.6)
            # Colour = gravity-weighted mix of 3 body colours
            c0 = bcolor[0]
            c1 = bcolor[1]
            c2 = bcolor[2]
            w  = ti.Vector([0.0, 0.0, 0.0])
            for j in range(3):
                d2 = (tpos[i] - bpos[j]).norm_sqr() + SOFTENING_SQ
                w[j] = 1.0 / d2
            wsum = w[0] + w[1] + w[2] + 1e-10
            tcol[i] = (c0 * w[0] + c1 * w[1] + c2 * w[2]) / wsum
            talive[i] = 1


@ti.kernel
def spawn_burst(wx: float, wy: float, seed: int):
    # Spawn burst of particles at a world position (right-click)
    for i in range(N_TEST):
        if talive[i] == 1:
            continue
        r   = 0.02 + ti.random(ti.f32) * 0.12
        ang = ti.random(ti.f32) * 6.2831853
        tpos[i]  = ti.Vector([wx + r * ti.cos(ang), wy + r * ti.sin(ang)])
        speed    = 0.3 + ti.random(ti.f32) * 0.4
        tang     = ti.Vector([-ti.sin(ang), ti.cos(ang)])
        tvel[i]  = tang * speed
        tcol[i]  = ti.Vector([0.9 + ti.random(ti.f32)*0.1,
                               0.8 + ti.random(ti.f32)*0.1,
                               0.7 + ti.random(ti.f32)*0.1])
        talive[i] = 1


@ti.kernel
def record_trail():
    # Write current body positions into ring-buffer trail
    for i in range(3):
        h = trail_head[i]
        trail_pos[i, h] = bpos[i]
        trail_head[i] = (h + 1) % TRAIL_LEN


@ti.kernel
def clear_pixels():
    # Fade pixel buffer slightly (ghost-trail effect) instead of full clear
    for px, py in pixels:
        old = pixels[px, py]
        pixels[px, py] = ti.Vector([old[0]*0.10, old[1]*0.10, old[2]*0.10, 1.0])


@ti.func
def world_to_screen_f(w):
    sx = int(w[0] * SCALE + CX)
    sy = int(w[1] * SCALE + CY)
    return sx, sy


@ti.func
def draw_glow(sx: int, sy: int, col: ti.types.vector(3, ti.f32), radius: float, brightness: float):
    # Paint a radial Gaussian glow spot into the HDR pixel buffer
    ri = int(radius) + 2
    for dx in range(-ri, ri+1):
        for dy in range(-ri, ri+1):
            px = sx + dx
            py = sy + dy
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                d2    = float(dx*dx + dy*dy)
                atten = brightness * ti.exp(-d2 / (radius*radius + 1e-4))
                existing = pixels[px, py]
                pixels[px, py] = ti.Vector([
                    ti.min(existing[0] + col[0]*atten, 3.0),
                    ti.min(existing[1] + col[1]*atten, 3.0),
                    ti.min(existing[2] + col[2]*atten, 3.0),
                    1.0
                ])


@ti.kernel
def render_trails():
    # Called only when trails are enabled (guarded from Python side)
    for bi in range(3):
        h   = trail_head[bi]
        col = bcolor[bi]
        for k in range(TRAIL_LEN):
            age = float(k) / TRAIL_LEN      # 0=oldest, 1=newest
            idx = (h + k) % TRAIL_LEN
            w   = trail_pos[bi, idx]
            sx, sy = world_to_screen_f(w)
            if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
                alpha = age * age * age      # cubic fade: trails dim near tail
                draw_glow(sx, sy, col, 2.5, alpha * 1.4)


@ti.kernel
def render_particles():
    for i in range(N_TEST):
        if talive[i] == 0:
            continue
        p  = tpos[i]
        sx, sy = world_to_screen_f(p)
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            col    = tcol[i]
            speed  = tvel[i].norm()
            bright = 0.3 + ti.min(speed * 0.6, 1.4)   # faster = brighter
            draw_glow(sx, sy, col, 1.8, bright)


@ti.kernel
def render_bodies():
    for i in range(3):
        p  = bpos[i]
        sx, sy = world_to_screen_f(p)
        col = bcolor[i]
        draw_glow(sx, sy, col, 18.0, 1.0)   # outer halo
        draw_glow(sx, sy, col,  8.0, 2.5)   # mid glow
        draw_glow(sx, sy, ti.Vector([1.0, 1.0, 1.0]), 3.0, 4.0)  # bright core


@ti.kernel
def render_ring(bx: float, by: float, bi: int, hill_r: float):
    # Dashed Hill-sphere ring around highlighted body
    col = bcolor[bi]
    for k in range(256):
        ang = float(k) / 256.0 * 6.2831853
        wx  = bx + hill_r * ti.cos(ang)
        wy  = by + hill_r * ti.sin(ang)
        sx  = int(wx * SCALE + CX)
        sy  = int(wy * SCALE + CY)
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            if (k // 8) % 2 == 0:
                draw_glow(sx, sy, col, 1.5, 0.5)


@ti.kernel
def render_field_lines():
    # Sample gravitational potential on a grid; dim dots mark equipotentials
    GRID = 40
    for gi in range(GRID * GRID):
        gx = gi % GRID
        gy = gi // GRID
        wx = (float(gx) / GRID - 0.5) * 8.0
        wy = (float(gy) / GRID - 0.5) * 5.0
        p  = ti.Vector([wx, wy])
        phi = 0.0
        for bi in range(3):
            d  = (p - bpos[bi]).norm() + 0.02
            phi -= G * bmass[bi] / d   # Phi = -G*m/r (Newtonian potential)
        bright = ti.min(-phi * 0.04, 0.25)
        sx = int(wx * SCALE + CX)
        sy = int(wy * SCALE + CY)
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            grey = ti.Vector([bright, bright, bright * 1.5])
            draw_glow(sx, sy, grey, 2.0, bright)


@ti.kernel
def tonemap():
    # Reinhard tone-mapping: compresses HDR [0,inf) to display [0,1)
    for px, py in pixels:
        c = pixels[px, py]
        r = c[0] / (1.0 + c[0])
        g = c[1] / (1.0 + c[1])
        b = c[2] / (1.0 + c[2])
        # Subtle saturation boost
        lum = 0.2126*r + 0.7152*g + 0.0722*b
        r   = lum + (r - lum) * 1.3
        g   = lum + (g - lum) * 1.3
        b   = lum + (b - lum) * 1.3
        pixels[px, py] = ti.Vector([ti.max(r,0.0), ti.max(g,0.0), ti.max(b,0.0), 1.0])


# --- Python-side helpers ---

def init_preset(name):
    cfg = PRESETS[name]
    for i in range(3):
        bpos[i]  = cfg[i][:2]
        bvel[i]  = cfg[i][2:]
        bmass[i] = BODY_MASS
    # Bootstrap leapfrog accelerations
    for i in range(3):
        a = [0.0, 0.0]
        for j in range(3):
            if i == j:
                continue
            rx = cfg[j][0] - cfg[i][0]
            ry = cfg[j][1] - cfg[i][1]
            d2 = rx*rx + ry*ry + SOFTENING_SQ
            d3 = math.sqrt(d2) * d2
            a[0] += G * BODY_MASS * rx / d3
            a[1] += G * BODY_MASS * ry / d3
        bacc[i] = a
    for i in range(3):
        trail_head[i] = 0
        for k in range(TRAIL_LEN):
            trail_pos[i, k] = cfg[i][:2]
    for i in range(N_TEST):
        talive[i] = 0


def compute_hill_radius(bi):
    p0 = bpos[bi].to_numpy()
    dmin = 1e9
    for j in range(3):
        if j == bi:
            continue
        p1 = bpos[j].to_numpy()
        d  = math.hypot(p0[0]-p1[0], p0[1]-p1[1])
        dmin = min(dmin, d)
    # Hill sphere: R_H = a * (m / 3M)^(1/3), simplified for equal masses
    return dmin * (1.0/3.0)**(1.0/3.0)


# --- Main loop ---

def main():
    preset_idx  = 0
    preset_name = PRESET_NAMES[preset_idx]
    init_preset(preset_name)
    respawn_particles(42)

    window = ti.ui.Window("Three-Body Problem", (WIDTH, HEIGHT),
                          fps_limit=120)
    canvas = window.get_canvas()
    gui    = window.get_gui()

    time_scale   = TIME_SCALE_INIT
    show_trails  = True
    highlight    = -1
    frame        = 0
    respawn_seed = 0
    t_sim        = 0.0
    fps_smooth   = 60.0

    prev_space = prev_r = prev_t = prev_f = False
    prev_1 = prev_2 = prev_3 = False
    prev_rmb = False

    while window.running:
        t0 = time.time()

        # -- Input --
        cur_space = window.is_pressed(ti.ui.SPACE)
        cur_r     = window.is_pressed('r')
        cur_t     = window.is_pressed('t')
        cur_f     = window.is_pressed('f')
        cur_plus  = window.is_pressed('=')
        cur_minus = window.is_pressed('-')
        cur_1     = window.is_pressed('1')
        cur_2     = window.is_pressed('2')
        cur_3     = window.is_pressed('3')
        cur_esc   = window.is_pressed(ti.ui.ESCAPE)

        if cur_esc:
            break

        if cur_space and not prev_space:
            paused[None] = 1 - paused[None]
            state = "PAUSED" if paused[None] else "RUNNING"
            print(state)
        if cur_r and not prev_r:
            init_preset(preset_name)
            respawn_particles(respawn_seed)
            t_sim = 0.0
            print("Reset:", preset_name)
        if cur_t and not prev_t:
            show_trails = not show_trails
            print("Trails:", "ON" if show_trails else "OFF")
        if cur_f and not prev_f:
            preset_idx  = (preset_idx + 1) % len(PRESET_NAMES)
            preset_name = PRESET_NAMES[preset_idx]
            init_preset(preset_name)
            respawn_particles(respawn_seed)
            t_sim = 0.0
            print("Switched to:", preset_name)
        if cur_1 and not prev_1:
            highlight = 0 if highlight != 0 else -1
        if cur_2 and not prev_2:
            highlight = 1 if highlight != 1 else -1
        if cur_3 and not prev_3:
            highlight = 2 if highlight != 2 else -1

        if cur_plus:
            time_scale = min(time_scale * 1.03, 5.0)
        if cur_minus:
            time_scale = max(time_scale * 0.97, 0.05)

        prev_space = cur_space
        prev_r = cur_r
        prev_t = cur_t
        prev_f = cur_f
        prev_1 = cur_1
        prev_2 = cur_2
        prev_3 = cur_3

        mx, my = window.get_cursor_pos()
        wx_m   = (mx * WIDTH  - CX) / SCALE
        wy_m   = (my * HEIGHT - CY) / SCALE
        mouse_world[None] = [wx_m, wy_m]

        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)
        mouse_pull[None] = 1.0 if lmb else 0.0

        if rmb and not prev_rmb:
            spawn_burst(wx_m, wy_m, frame)
        prev_rmb = rmb

        # -- Physics --
        if paused[None] == 0:
            dt_eff = DT_BASE * time_scale
            dt_sub = dt_eff / SUBSTEPS
            for _ in range(SUBSTEPS):
                integrate_bodies(dt_sub)
                integrate_particles(dt_sub)
            t_sim += dt_eff
            if frame % 4 == 0:
                respawn_seed += 1
                respawn_particles(respawn_seed)
            if frame % 2 == 0:
                record_trail()

        # -- Render --
        clear_pixels()
        if frame % 4 == 0:
            render_field_lines()
        if show_trails:
            render_trails()
        render_particles()
        render_bodies()
        if highlight >= 0:
            hr = compute_hill_radius(highlight)
            bp = bpos[highlight].to_numpy()
            render_ring(bp[0], bp[1], highlight, hr)
        tonemap()
        canvas.set_image(pixels)

        # -- HUD --
        dt_frame = time.time() - t0
        fps_smooth = fps_smooth * 0.95 + (1.0 / max(dt_frame, 1e-5)) * 0.05
        with gui.sub_window("Info", 0.01, 0.01, 0.26, 0.20):
            gui.text("Preset: " + preset_name)
            gui.text("Time x" + str(round(time_scale, 2)) + "  T=" + str(round(t_sim, 2)))
            gui.text("FPS: " + str(int(fps_smooth)))
            gui.text("Particles: " + str(N_TEST))
            gui.text("PAUSED" if paused[None] else "RUNNING")

        window.show()
        frame += 1

    window.destroy()


if __name__ == "__main__":
    main()
