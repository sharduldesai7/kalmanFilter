import unittest
from kf import KF
import numpy as np

class TestKF(unittest.TestCase):
    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 0.8

        kf = KF(x,v, accel_variance=1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)

    def test_can_predict_and_mean_and_cov_are_of_right_shape(self):
        x = 0.2
        v = 0.8

        kf = KF(x,v, accel_variance=1.2)
        kf.predict(dt=0.1)

        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2,))

    def test_calling_predict_increases_state_uncertainty(self):
        x = 0.2
        v = 0.8

        kf = KF(x,v, accel_variance=1.2)

        for i in range(10):
            det_cov_before = np.linalg.det(kf.cov)
            #print(det_cov_before)
            #print("---")
            kf.predict(dt=0.1)
            det_cov_after = np.linalg.det(kf.cov)
            #print(det_cov_after)
            #print(" ")

            self.assertGreater(det_cov_after, det_cov_before)

    def test_calling_update_does_not_crash(self):
        x = 0.2
        v = 0.8

        kf = KF(x,v, accel_variance=1.2)

        meas_value = 0.1
        meas_variance = 0.1
        kf.update(meas_value, meas_variance)

    def test_calling_update_decreases_state_uncertainty(self):
        x = 0.2
        v = 0.8
        meas_value = 0.1
        meas_variance = 0.1

        kf = KF(x,v, accel_variance=1.2)

        for i in range(10):
            det_cov_before = np.linalg.det(kf.cov)
            #print(det_cov_before)
            #print("---")
            kf.update(meas_value, meas_variance)
            det_cov_after = np.linalg.det(kf.cov)
            #print(det_cov_after)
            #print(" ")

            self.assertLess(det_cov_after, det_cov_before)




