import taichi as ti
import math

ti.init(arch = ti.cpu)

# constants
dt = 1e-3
substeps = 10
g = 9.8
res = 500
max_points_stored = 10000
# variables
sim = ti.field(ti.int32, ())
traj_b_enabled = ti.field(ti.int32, ())

origin = ti.Vector.field(2, ti.f32, 1)
pos_a = ti.Vector.field(2, ti.f32, 1)
pos_b = ti.Vector.field(2, ti.f32, 1)
L = ti.field(ti.f32, 2)
m = ti.field(ti.f32, 2)
theta = ti.field(ti.f32, 2)
omega = ti.field(ti.f32, 2)
domega = ti.field(ti.f32, 2)
E = ti.field(ti.f32, ())
E_init = ti.field(ti.f32, ())

num_p = ti.field(ti.i32, ())
traj_b = ti.Vector.field(2, ti.f32, max_points_stored)

@ti.func
def compute_pos():
    x0 = origin[0][0] + L[0] * ti.sin(theta[0])
    y0 = origin[0][1] - L[0] * ti.cos(theta[0])
    x1 = x0 + L[1] * ti.sin(theta[1])
    y1 = y0 - L[1] * ti.cos(theta[1])
    pos_a[0] = ti.Vector([x0, y0])
    pos_b[0] = ti.Vector([x1, y1])

@ti.func
def compute_E():
    T = 0.5 * (m[0]+m[1]) * (omega[0] * L[0])**2 + 0.5 * m[1] * (omega[1]*L[1])**2 + \
        m[1]*L[0]*L[1]* ti.cos(theta[0] - theta[1]) * omega[0] * omega[1]
    V = (m[0]+m[1])*g*L[0]*ti.cos(theta[0]) + m[1]*g*L[1]*ti.cos(theta[1])
    E[None] = T + V

@ti.kernel
def initialize():
    sim = 0
    traj_b_enabled = 0
    # mid point of the top
    origin[0] = ti.Vector([0.5, 1.0])
    # length of two pendulums
    L[0] = 0.25
    L[1] = 0.25
    # mass of two balls
    m[0] = 50.0
    m[1] = 50.0
    # initial angels
    theta[0] = math.pi / 4.0
    theta[1] = math.pi / 2.0
    # initial angular velocity
    omega[0] = 0.0
    omega[1] = 0.0
    # compute initial positions
    compute_pos()
    traj_b[0] = pos_b[0]
    num_p[None] = 1
    # compute inital energy
    compute_E()
    E_init[None] = E[None]

@ti.func
def compute_domega():
    s01 = ti.sin(theta[0] - theta[1])
    c01 = ti.cos(theta[0] - theta[1])
    s0 = ti.sin(theta[0])
    s1 = ti.sin(theta[1])
    o0_sqr = omega[0] * omega[0]
    o1_sqr = omega[1] * omega[1]

    denom = m[1] * c01 - m[0] - m[1]

    domega[0] = (m[1]*L[1]*s01*o1_sqr + m[1]*L[0]*c01*s01*o0_sqr + \
                (m[0]+m[1])*g*s0 - m[1]*g*s1*c01) / (L[0] * denom)
    domega[1] = (m[1]*L[1]*s01*c01*o1_sqr + (m[0]+m[1])*L[0]*s01*o0_sqr + \
                (m[0]+m[1])*g*(s0*c01 - s1) ) / (-L[1] * denom)

@ti.kernel
def update():
    for i in range(substeps):
        compute_domega()
        omega[0] += dt * domega[0]
        omega[1] += dt * domega[1]
        theta[0] += dt * omega[0]
        theta[1] += dt * omega[1]
        compute_pos()
    i = num_p[None]
    if i<max_points_stored:
        traj_b[i] = pos_b[0]
        num_p[None] += 1
    # check total energy
    compute_E()
    delta_E = (E[None] - E_init[None]) / E_init[None]
    #print('Total energy percentage change is ', delta_E*100)

def main():
    gui = ti.GUI('double pendulums', (res, res))
    initialize()
    print('initial angle 1 is ', int(theta[0] * 180 / math.pi), 'degrees')
    print('initial angle 2 is ', int(theta[1] * 180 / math.pi), 'degrees')
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == 's':
                sim[None] = not sim[None]
            elif e.key == 't':
                traj_b_enabled[None] = not traj_b_enabled[None]
        if sim[None]:
            update()
        
        gui.line(begin = origin[0], end = pos_a[0], color = 0xffffff)
        gui.circle(pos_a[0], color = 0xff0000, radius = 10)
        gui.line(begin = pos_a[0], end = pos_b[0], color = 0xffffff)
        gui.circle(pos_b[0], color = 0x0000ff, radius = 10)
        if traj_b_enabled[None]:
            gui.lines(begin = traj_b.to_numpy()[0:num_p[None]-1], end=traj_b.to_numpy()[1:num_p[None]], color = 0x0000ff)
        gui.show()

if __name__ == '__main__':
    main()