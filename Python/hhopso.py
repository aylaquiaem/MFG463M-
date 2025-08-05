import numpy as np
import matplotlib.pyplot as plt

def HHOPSO(func, dim, bounds, steps, size, w, c1, c2, seed=0, vis=False):
    np.random.seed(seed)

    lb = bounds[:, 0]
    ub = bounds[:, 1]

    bestsol = np.zeros(dim)
    bestfit = np.inf
    hist = np.zeros(steps)

    # Initialize population and velocity
    pop = lb + np.random.rand(size, dim) * (ub - lb)
    fit = np.array([func(ind) for ind in pop])
    range_ = (ub - lb) * 0.5
    vel = (2 * np.random.rand(size, dim) - 1) * range_
    sbpop = pop.copy()
    sbfit = fit.copy()

    # Visualization setup
    if dim == 2 and vis:
        fig, ax = plt.subplots()
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[func([xi, yi]) for xi in x] for yi in y])
        contour = ax.contour(X, Y, Z, levels=30, linewidths=1.5)
        plt.colorbar(contour, ax=ax)
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.set_aspect('equal')

        h = ax.scatter(pop[:, 0], pop[:, 1], c='red', s=20, label='Agents')
        h_best = ax.scatter(bestsol[0], bestsol[1], c='black', s=40, label='Best')
        ax.legend()
        plt.ion()
        plt.show()

    # Main loop
    for t in range(steps):
        for i in range(size):
            if fit[i] < sbfit[i]:
                sbpop[i] = pop[i].copy()
                sbfit[i] = fit[i]

            if sbfit[i] < bestfit:
                bestsol = sbpop[i].copy()
                bestfit = sbfit[i]

        hist[t] = bestfit
        popM = np.mean(pop, axis=0)

        for i in range(size):
            popR = pop[np.random.randint(size)]
            r1, r2, r3, r4 = np.random.rand(4)

            Eo = 2 * np.random.rand() - 1
            E = 2 * Eo * (1 - t / steps)

            if abs(E) >= 1:
                # Exploration (HHO)
                q = np.random.rand()
                if q >= 0.5:
                    pop[i] = popR - r1 * np.abs(popR - 2 * r2 * pop[i])
                else:
                    pop[i] = (bestsol - popM) - r3 * (lb + r4 * (ub - lb))
                pop[i] = np.clip(pop[i], lb, ub)
                fit[i] = func(pop[i])
            else:
                # Exploitation (PSO)
                vel[i] = (
                    w * vel[i]
                    + c1 * np.random.rand() * (sbpop[i] - pop[i])
                    + c2 * np.random.rand() * (bestsol - pop[i])
                )
                pop[i] += vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                fit[i] = func(pop[i])

        # Visualization update
        if dim == 2 and vis:
            h.set_offsets(pop[:, :2])
            h_best.set_offsets(bestsol[:2])
            ax.set_title(f"HHO Step {t+1} | Best: {bestfit:.12f}")
            plt.pause(0.01)

    if dim == 2 and vis:
        plt.ioff()
        plt.show()

    return bestsol, bestfit, hist