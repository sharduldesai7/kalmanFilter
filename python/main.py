import numpy as np
import matplotlib.pyplot as plt

from kf import KF

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEPS = 20

plt.ion()
plt.figure()

real_x = 0.0
meas_variance = 0.1 ** 2
real_v = 0.9

kf = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1)

mus = []
covs = []

real_xs = []
real_vs = []

for i in range(NUM_STEPS):
    if NUM_STEPS - i < 500:
        real_v -= 0.01

    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_v

    kf.predict(dt=DT)
    if i != 0 and i % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value = real_x + np.random.randn() * np.sqrt(meas_variance),
                  meas_variance = meas_variance)        

    real_xs.append(real_x)
    real_vs.append(real_v)

plt.subplot(2,1,1)
plt.title("Position")
plt.plot([mu[0] for mu in mus], 'r')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus, covs)], 'r--')

plt.subplot(2,1,2)
plt.title("Velocity")
plt.plot([mu[1] for mu in mus], 'r')
plt.plot(real_vs, 'b')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus, covs)], 'r--')

plt.show()
plt.ginput(1)
