# %%

import numpy as np
from scipy.special import comb

# %%


def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [ [1,1],
              [2,3],
              [4,5], ..[Xn, Yn] ]
     nTimes is the number of time steps, defaults to 1000

     See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)]
    )

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return np.stack([xvals, yvals])


# %%
from matplotlib import pyplot as plt

# nPoints = 4
# points = np.random.rand(nPoints,2)*200
xpoints = [-1, -1, -100, 100, 1, 1]
ypoints = [0, 5, 5, 5, 5, 0]
points = np.array([xpoints, ypoints]).T
# xpoints = [p[0] for p in points]
# ypoints = [p[1] for p in points]

curve = bezier_curve(points, nTimes=1000)
xvals, yvals = curve
plt.plot(*curve)
# plt.plot(xpoints, ypoints, "ro")
# for nr in range(len(points)):
#     plt.text(points[nr][0], points[nr][1], nr)

plt.show()
# %%
np.sqrt(((curve[:, :-1] - curve[:, 1:]) ** 2).sum(0)).sum()
# %%
