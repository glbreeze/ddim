import numpy as np
from scipy.linalg import qr, eigvals
from seaborn import distplot

n = 10
repeats = 2

angles = []
angles_modified = []
for rp in range(repeats):
    H = np.random.randn(n, n)
    Q, R = qr(H)
    angles.append(np.angle(eigvals(Q)))
    Q_modified = Q @ np.diag(np.exp(1j * np.pi * 2 * np.random.rand(n)))
    angles_modified.append(np.angle(eigvals(Q_modified)))

fig, ax = plt.subplots(1,2, figsize = (10,3))
distplot(np.asarray(angles).flatten(),kde = False, hist_kws=dict(edgecolor="k", linewidth=2), ax= ax[0])
ax[0].set(xlabel='phase', title='direct')
distplot(np.asarray(angles_modified).flatten(),kde = False, hist_kws=dict(edgecolor="k", linewidth=2), ax= ax[1])
ax[1].set(xlabel='phase', title='modified');

R = Q[:, :5] @ Q[:, :5].T - Q[:, 5:] @ Q[:, 5:].T