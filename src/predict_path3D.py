import numpy as np

class KalmanFilter:
    """
    State: [p1..pn, v1..vn]^T  (n = dims)
    Measurement: positions only (n dims)
    Adaptive R (Mahalanobis gating) + adaptive Q (maneuvers), with warm-up.
    """
    def __init__(self, dt=0.05, sigma_a=1.0, sigma_pos=5.0, dims=3, rz_scale=4.0,
                 init_pos_var=9.0, init_vel_var=25.0, warmup_updates=5):
        self.dims = int(dims)
        self.dt = float(dt)
        self.sigma_a = float(sigma_a)
        self.sigma_pos = float(sigma_pos)

        # Build model matrices
        self._build_F_H()
        self._build_Q_base()
        self._build_R_base(rz_scale=rz_scale)

        # State and covariance
        self.x = np.zeros((2*self.dims, 1), dtype=float)
        self.P = np.eye(2*self.dims, dtype=float)
        self.P[:self.dims, :self.dims] *= float(init_pos_var)
        self.P[self.dims:, self.dims:] *= float(init_vel_var)

        # Adaptive scalers
        self.r_scale = 1.0
        self.q_scale = 1.0
        self.adapt_alpha = 0.2  

        # Gating parameters
        self.chi2_gate = {2: 9.21, 3: 11.34}.get(self.dims, 9.21)
        self.R_scale_hi = 25.0
        self.Q_scale_lo = 0.5
        self.Q_scale_hi = 3.0

        # Warm-up disables aggressive adaptation initially
        self.warmup_updates = int(max(0, warmup_updates))
        self.updates = 0
        self.initialized = False

    def _build_F_H(self):
        n = self.dims
        I = np.eye(n)
        Z = np.zeros((n, n))
        self.F = np.block([[I, self.dt * I],
                           [Z,           I]])
        self.H = np.hstack([I, Z])  # positions only

    def _build_Q_base(self):
        dt = self.dt
        q11 = (dt**4) / 4.0
        q12 = (dt**3) / 2.0
        q22 = (dt**2)
        Q_dim = np.array([[q11, q12],
                          [q12, q22]], dtype=float) * (self.sigma_a**2)
        blocks = [Q_dim for _ in range(self.dims)]
        self.Q_base = self._blkdiag(blocks)

    def _build_R_base(self, rz_scale=4.0):
        R = np.eye(self.dims, dtype=float) * (self.sigma_pos**2)
        if self.dims >= 3 and rz_scale is not None:
            R[2, 2] *= float(rz_scale)  # distance/no-depth noisier
        self.R_base = R

    @staticmethod
    def _blkdiag(blocks):
        nrows = sum(b.shape[0] for b in blocks)
        ncols = sum(b.shape[1] for b in blocks)
        out = np.zeros((nrows, ncols), dtype=float)
        r = c = 0
        for b in blocks:
            rr, cc = b.shape
            out[r:r+rr, c:c+cc] = b
            r += rr
            c += cc
        return out

    def set_dt(self, dt):
        self.dt = float(dt)
        self._build_F_H()
        self._build_Q_base()  # rebuild base Q for new dt

    def initialize(self, z, v0=None):
        z = np.asarray(z, dtype=float).reshape(self.dims, 1)
        self.x[:self.dims, 0:1] = z
        if v0 is None:
            self.x[self.dims:, 0:1] = 0.0
        else:
            v0 = np.asarray(v0, dtype=float).reshape(self.dims, 1)
            self.x[self.dims:, 0:1] = v0
        self.r_scale = 1.0
        self.q_scale = 1.0
        self.updates = 0
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def get_velocity(self):
        return self.x[self.dims:, 0]

    def _adaptive_R_Q(self, y, P_pred):
        # During warm-up: no gating, nominal R/Q
        if self.updates < self.warmup_updates:
            return self.R_base, self.Q_base

        S_nom = self.H @ P_pred @ self.H.T + self.R_base
        try:
            S_inv = np.linalg.inv(S_nom)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S_nom)

        d2 = float(y.T @ S_inv @ y)  # Mahalanobis distance^2

        # Adaptive R: inflate on outliers
        target_r = self.R_scale_hi if d2 > self.chi2_gate else 1.0
        self.r_scale = (1 - self.adapt_alpha) * self.r_scale + self.adapt_alpha * target_r
        R_use = self.R_base * self.r_scale

        # Adaptive Q: allow more dynamics when innovation grows
        raw_q = 0.3 + 0.1 * d2
        target_q = float(np.clip(raw_q, self.Q_scale_lo, self.Q_scale_hi))
        self.q_scale = (1 - self.adapt_alpha) * self.q_scale + self.adapt_alpha * target_q
        Q_use = self.Q_base * self.q_scale
        return R_use, Q_use

    def predict(self):
        self.P = self.F @ self.P @ self.F.T + (self.Q_base * self.q_scale)
        self.x = self.F @ self.x
        pos = self.x[:self.dims].copy()
        vel = self.x[self.dims:].copy()
        return pos, vel

    def update(self, z, R_override=None):
        z = np.asarray(z, dtype=float).reshape(self.dims, 1)

        # Initialize on first measurement
        if not self.initialized:
            self.initialize(z)
            return

        x_pred = self.x
        P_pred = self.P
        y = z - (self.H @ x_pred)

        if R_override is None:
            R_use, Q_use = self._adaptive_R_Q(y, P_pred)
        else:
            R_use = R_override
            Q_use = self.Q_base * self.q_scale

        S = self.H @ P_pred @ self.H.T + R_use
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_pred @ self.H.T @ S_inv
        self.x = x_pred + K @ y
        I = np.eye(self.P.shape[0], dtype=float)
        self.P = (I - K @ self.H) @ P_pred

        # Store effective Q for next predict via q_scale
        self.Q = Q_use
        self.updates += 1

