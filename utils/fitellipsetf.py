import math
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from warnings import warn
from scipy import optimize, spatial

_EPSILON = np.spacing(1)
NUMBER = 256
#_EPSILON = tf.keras.backend.epsilon()


def _check_data_dim(data, dim):
    #if data.ndim != 2 or data.shape[1] != dim:
    if len(data.shape) != 2 or data.shape[1] != dim:
        raise ValueError(f"Input data must have shape (N, {dim}).")


def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError('Input data must be at least 2D.')


class BaseModel:

    def __init__(self):
        self.params = None


class LineModelND(BaseModel):
    """Total least squares estimator for N-dimensional lines.

    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.

    Lines are defined by a point (origin) and a unit vector (direction)
    according to the following vector equation::

        X = origin + lambda * direction

    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `direction`.

    Examples
    --------
    >>> x = tnp.linspace(1, 2, 25)
    >>> y = 1.5 * x + 3
    >>> lm = LineModelND()
    >>> lm.estimate(tnp.stack([x, y], axis=-1))
    True
    >>> tuple(tnp.round(lm.params, 5))
    (array([1.5 , 5.25]), array([0.5547 , 0.83205]))
    >>> res = lm.residuals(tnp.stack([x, y], axis=-1))
    >>> tnp.abs(tnp.round(res, 9))
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    >>> tnp.round(lm.predict_y(x[:5]), 3)
    array([4.5  , 4.562, 4.625, 4.688, 4.75 ])
    >>> tnp.round(lm.predict_x(y[:5]), 3)
    array([1.   , 1.042, 1.083, 1.125, 1.167])

    """

    def estimate(self, data):
        """Estimate line model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        _check_data_atleast_2D(data)

        origin = data.mean(axis=0)
        data = data - origin

        if data.shape[0] == 2:  # well determined
            direction = data[1] - data[0]
            norm = tnp.linalg.norm(direction)
            if norm != 0:  # this should not happen to be norm 0
                direction /= norm
        elif data.shape[0] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _, _, v = tnp.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:  # under-determined
            return False

        self.params = (origin, direction)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        _check_data_atleast_2D(data)
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params
        res = (data - origin) - \
              ((data - origin) @ direction)[..., tnp.newaxis] * direction
        return tnp.linalg.norm(res, axis=1)

    def predict(self, x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.

        Parameters
        ----------
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        data : (n, m) array
            Predicted coordinates.

        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params

        if direction[axis] == 0:
            # line parallel to axis
            raise ValueError(f'Line parallel to axis {axis}')

        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., tnp.newaxis] * direction
        return data

    def predict_x(self, y, params=None):
        """Predict x-coordinates for 2D lines using the estimated model.

        Alias for::

            predict(y, axis=1)[:, 0]

        Parameters
        ----------
        y : array
            y-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        x : array
            Predicted x-coordinates.

        """
        x = self.predict(y, axis=1, params=params)[:, 0]
        return x

    def predict_y(self, x, params=None):
        """Predict y-coordinates for 2D lines using the estimated model.

        Alias for::

            predict(x, axis=0)[:, 1]

        Parameters
        ----------
        x : array
            x-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        y : array
            Predicted y-coordinates.

        """
        y = self.predict(x, axis=0, params=params)[:, 1]
        return y


class CircleModel(BaseModel):

    """Total least squares estimator for 2D circles.

    The functional model of the circle is::

        r**2 = (x - xc)**2 + (y - yc)**2

    This estimator minimizes the squared distances from all points to the
    circle::

        min{ sum((r - sqrt((x_i - xc)**2 + (y_i - yc)**2))**2) }

    A minimum number of 3 points is required to solve for the parameters.

    Attributes
    ----------
    params : tuple
        Circle model parameters in the following order `xc`, `yc`, `r`.

    Notes
    -----
    The estimation is carried out using a 2D version of the spherical
    estimation given in [1]_.

    References
    ----------
    .. [1] Jekel, Charles F. Obtaining non-linear orthotropic material models
           for pvc-coated polyester via inverse bubble inflation.
           Thesis (MEng), Stellenbosch University, 2016. Appendix A, pp. 83-87.
           https://hdl.handle.net/10019.1/98627

    Examples
    --------
    >>> t = tnp.linspace(0, 2 * tnp.pi, 25)
    >>> xy = CircleModel().predict_xy(t, params=(2, 3, 4))
    >>> model = CircleModel()
    >>> model.estimate(xy)
    True
    >>> tuple(tnp.round(model.params, 5))
    (2.0, 3.0, 4.0)
    >>> res = model.residuals(xy)
    >>> tnp.abs(tnp.round(res, 9))
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def estimate(self, data):
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        _check_data_dim(data, dim=2)

        # to prevent integer overflow, cast data to float, if it isn't already
        float_type = tnp.promote_types(data.dtype, tnp.float32)
        data = data.astype(float_type, copy=False)

        # Adapted from a spherical estimator covered in a blog post by Charles
        # Jeckel (see also reference 1 above):
        # https://jekel.me/2015/Least-Squares-Sphere-Fit/
        A = tnp.append(data * 2,
                      tnp.ones((data.shape[0], 1), dtype=float_type),
                      axis=1)
        f = tnp.sum(data ** 2, axis=1)
        C, _, rank, _ = tnp.linalg.lstsq(A, f, rcond=None)

        if rank != 3:
            warn("Input does not contain enough significant data points.")
            return False

        center = C[0:2]
        distances = spatial.minkowski_distance(center, data)
        r = tnp.sqrt(tnp.mean(distances ** 2))

        self.params = tuple(center) + (r,)

        return True

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the circle is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        """

        _check_data_dim(data, dim=2)

        xc, yc, r = self.params

        x = data[:, 0]
        y = data[:, 1]

        return r - tnp.sqrt((x - xc)**2 + (y - yc)**2)

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (3, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """
        if params is None:
            params = self.params
        xc, yc, r = params

        x = xc + r * tnp.cos(t)
        y = yc + r * tnp.sin(t)

        return tnp.concatenate((x[..., None], y[..., None]), axis=t.ndim)


class EllipseModel(BaseModel):
    """Total least squares estimator for 2D ellipses.

    The functional model of the ellipse is::

        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)

    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.

    The estimator is based on a least squares minimization. The optimal
    solution is computed directly, no iterations are required. This leads
    to a simple, stable and robust fitting method.

    The ``params`` attribute contains the parameters in the following order::

        xc, yc, a, b, theta

    Attributes
    ----------
    params : tuple
        Ellipse model parameters in the following order `xc`, `yc`, `a`, `b`,
        `theta`.

    Examples
    --------

    >>> xy = EllipseModel().predict_xy(tnp.linspace(0, 2 * tnp.pi, 25),
    ...                                params=(10, 15, 4, 8, tnp.deg2rad(30)))
    >>> ellipse = EllipseModel()
    >>> ellipse.estimate(xy)
    True
    >>> tnp.round(ellipse.params, 2)
    array([10.  , 15.  ,  4.  ,  8.  ,  0.52])
    >>> tnp.round(abs(ellipse.residuals(xy)), 5)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    """
    def __init__(self):
        self.params = None

    # Based on np.linalg.cond(x, p=None)
    def tf_cond(self, x):
        x = tf.convert_to_tensor(x)
        s = tf.linalg.svd(x, compute_uv=False)
        r = s[..., 0] / s[..., -1]
        # Replace NaNs in r with infinite unless there were NaNs before
        x_nan = tf.reduce_any(tf.math.is_nan(x), axis=(-2, -1))
        r_nan = tf.math.is_nan(r)
        r_inf = tf.fill(tf.shape(r), tf.constant(math.inf, r.dtype))
        tf.where(x_nan, r, tf.where(r_nan, r_inf, r))
        return r

    def is_invertible(self, x, epsilon=1e-6):  # Epsilon may be smaller with tf.float64
        x = tf.convert_to_tensor(x)
        eps_inv = tf.cast(1 / epsilon, x.dtype)
        x_cond = self.tf_cond(x)
        return tf.math.is_finite(x_cond) & (x_cond < eps_inv)
    def pad_up_to(self, t, max_in_dims, constant_values):
        diff = max_in_dims - tf.shape(t)
        paddings = tf.pad(diff[:, None], [[0, 0], [1, 0]])
        return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)
    def estimate(self, data):
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.


        References
        ----------
        .. [1] Halir, R.; Flusser, J. "Numerically stable direct least squares
               fitting of ellipses". In Proc. 6th International Conference in
               Central Europe on Computer Graphics and Visualization.
               WSCG (Vol. 98, pp. 125-132).

        """


        # Original Implementation: Ben Hammel, Nick Sullivan-Molina
        # another REFERENCE: [2] http://mathworld.wolfram.com/Ellipse.html
        _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]

        # Quadratic part of design matrix [eqn. 15] from [1]
        #D1 = tnp.vstack([x ** 2, x * y, y ** 2]).T
        D1 = tf.transpose(tf.stack([x ** 2, x * y, y ** 2], axis=0))
        # Linear part of design matrix [eqn. 16] from [1]
        #D2 = tnp.vstack([x, y, tnp.ones_like(x)]).T
        D2 = tf.transpose(tf.stack([x, y, tnp.ones_like(x)],axis=0))

        # forming scatter matrix [eqn. 17] from [1]
        S1 = tf.transpose(D1) @ D1
        S2 = tf.transpose(D1) @ D2
        S3 = tf.transpose(D2) @ D2
        # Constraint matrix [eqn. 18]
        C1 = tf.constant([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]], dtype=tf.float64)

        # detc = tf.linalg.det(C1)
        # dets = tf.linalg.det(S3)
        # tf.print("dets:", dets)
        # tf.print("detc:", detc)
        # tf.print("s:", S3)
        # tf.print("c:", C1)


        # manual check
        if(self.is_invertible(S3) and self.is_invertible(C1)):
        #if tf.abs(detc) != 0 and tf.abs(dets) != 0:
        #if (tf.abs(detc) > 1e-5 or tf.abs(detc) < -1e-5) \
           #and (tf.abs(dets) > 1e-5 or tf.abs(dets) < -1e-5):
            #tf.print("Matrix is invertible")
            #print("Matrix is  invertible")

        
            try:
                # Reduced scatter matrix [eqn. 29]
                M = tf.linalg.inv(C1) @ (S1 - S2 @ tf.linalg.inv(S3) @ tf.transpose(S2))

            #except np.linalg.LinAlgError:  # LinAlgError: Singular matrix
            except tf.errors.InvalidArgumentError as e:
                # handle the InvalidArgumentError exception here
                tf.print("Invalid argument:", e)
            except StopIteration:
                tf.print("StopIteration")
            eig_vals, eig_vecs = tf.linalg.eig(M)
            eig_vecs = tf.math.real(eig_vecs)

            # eigenvector must meet constraint 4ac - b^2 to be valid.
            cond = 4 * tf.multiply(eig_vecs[0, :], eig_vecs[2, :]) - tf.math.pow(eig_vecs[1, :], 2)
            bcond = tf.math.greater(tf.math.real(cond), 0)

            a1 = tf.math.real(tf.boolean_mask(eig_vecs, bcond))
            # seeks for empty matrix
            if 0 in a1.shape or len(tf.reshape(a1,[-1])) != 3:
                params = tf.constant([0., 0., 0., 0., 0.], dtype=tf.float64)
                return params
            else:

                rv1 = tf.reshape(a1,[-1])
                a, b, c = rv1[0], rv1[1], rv1[2]
                a1_padded = self.pad_up_to(a1, [3, 3], 0)

                a2 = -tf.linalg.inv(S3) @ tf.transpose(S2) @ a1_padded

                rv2 = tf.reshape(a2,[-1])
                d, f, g = rv2[0], rv2[1], rv2[2]

                # eigenvectors are the coefficients of an ellipse in general form
                # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 (eqn. 15) from [2]
                b /= 2.
                d /= 2.
                f /= 2.

                #tf.print("xparams",self.xparams)

                # finding center of ellipse [eqn.19 and 20] from [2]
                x0 = (c * d - b * f) / (b ** 2. - a * c)
                y0 = (a * f - b * d) / (b ** 2. - a * c)

                # Find the semi-axes lengths [eqn. 21 and 22] from [2]
                numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g
                term = tf.math.sqrt((a - c) ** 2 + 4 * b ** 2)

                denominator1 = (b ** 2 - a * c) * (term - (a + c))
                denominator2 = (b ** 2 - a * c) * (- term - (a + c))

                width = tf.math.sqrt(2 * numerator / denominator1)
                height = tf.math.sqrt(2 * numerator / denominator2)

                # angle of counterclockwise rotation of major-axis of ellipse
                # to x-axis [eqn. 23] from [2].
                a,b,c = tf.math.real(a), tf.math.real(b), tf.math.real(c)
                phi = 0.5 * tf.math.atan((2. * b) / (a - c))
                if a > c:
                    phi += 0.5 * math.pi

                #self.params = tnp.nan_to_num([x0, y0, width, height, phi]).tolist()
                x0, y0, width, height, phi = tf.math.real(x0), tf.math.real(y0), tf.math.real(width), tf.math.real(height), tf.math.real(phi)
                w = [x0, y0, width, height, phi]
                #tf.print("w", w)

                params = tf.math.real(tf.where(tf.math.is_nan(w), tf.ones_like(w) * NUMBER, w))

                #tf.print("params", params)
                return  params

        else:
            params = tf.constant([0.,0.,0.,0.,0.], dtype=tf.float64)
            return params

        params = tf.constant([0.,0.,0.,0.,0.], dtype=tf.float64)
        return params
        tf.print("end of fit")

    def estimate0(self, data):
        self.params = [0.,10.,10.,10.,0.]
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.


        References
        ----------
        .. [1] Halir, R.; Flusser, J. "Numerically stable direct least squares
               fitting of ellipses". In Proc. 6th International Conference in
               Central Europe on Computer Graphics and Visualization.
               WSCG (Vol. 98, pp. 125-132).

        """

        # Original Implementation: Ben Hammel, Nick Sullivan-Molina
        # another REFERENCE: [2] http://mathworld.wolfram.com/Ellipse.html
        _check_data_dim(data, dim=2)

        # to prevent integer overflow, cast data to float, if it isn't already

        # float_type = tnp.promote_types(data.dtype, tnp.float32)
        # float_type = tf.float32
        # data = data.astype(float_type)  # , copy=False)

        x = data[:, 0]
        y = data[:, 1]

        # Quadratic part of design matrix [eqn. 15] from [1]
        #D1 = tnp.vstack([x ** 2, x * y, y ** 2]).T
        D1 = tf.transpose(tf.stack([x ** 2, x * y, y ** 2], axis=0))
        # Linear part of design matrix [eqn. 16] from [1]
        #D2 = tnp.vstack([x, y, tnp.ones_like(x)]).T
        D2 = tf.transpose(tf.stack([x, y, tnp.ones_like(x)],axis=0))

        # forming scatter matrix [eqn. 17] from [1]
        # S1 = D1.T @ D1
        # S2 = D1.T @ D2
        # S3 = D2.T @ D2
        S1 = tf.transpose(D1) @ D1
        S2 = tf.transpose(D1) @ D2
        S3 = tf.transpose(D2) @ D2

        # Constraint matrix [eqn. 18]
        C1 = tf.constant([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

        detc = tf.linalg.det(C1)
        dets = tf.linalg.det(S3)

        #if tf.abs(detc) != 0 or tf.abs(dets) != 0:
        if tf.abs(detc) == 0 or tf.abs(dets) == 0:
            #tf.print("Matrix is not invertible")
            return False

        #print("Matrix is  invertible")
        try:
            cinv = tf.linalg.inv(C1)
            #manual check
            # Reduced scatter matrix [eqn. 29]
            #M = tf.linalg.inv(C1) @ (S1 - S2 @ tf.linalg.inv(S3) @ S2.T)
            M = tf.linalg.inv(C1) @ (S1 - S2 @ tf.linalg.inv(S3) @ tf.transpose(S2))

        #except np.linalg.LinAlgError:  # LinAlgError: Singular matrix
        except tf.errors.InvalidArgumentError as e:
            # handle the InvalidArgumentError exception here
            tf.print("Invalid argument:", e)
            return False
        except StopIteration:
            return False

        eig_vals, eig_vecs = tf.linalg.eig(M)
        eig_vecs = tf.math.real(eig_vecs)
        eig_vals = tf.math.real(eig_vals)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * tf.multiply(eig_vecs[0, :], eig_vecs[2, :]) - tf.math.pow(eig_vecs[1, :], 2)
        bcond = tf.math.greater(tf.math.real(cond), 0)
        # a1 = eig_vecs[:, (cond > 0)]

        #a1 = eig_vecs[:, bcond]
        a1 = tf.math.real(tf.boolean_mask(eig_vecs, bcond))
        # seeks for empty matrix
        #if 0 in a1.shape or len(a1.ravel()) != 3:
        if 0 in a1.shape or len(tf.reshape(a1,[-1])) != 3:
            return False

        #rv1 = a1.ravel()
        rv1 = tf.reshape(a1,[-1])
        a, b, c = rv1[0], rv1[1], rv1[2]

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a1_padded = self.pad_up_to(a1, [3, 3], 0)

        #a2 = -tf.linalg.inv(S3) @ S2.T @ a1
        #a2 = -tf.linalg.inv(S3) @ tf.transpose(S2) @ a1
        a2 = -tf.linalg.inv(S3) @ tf.transpose(S2) @ a1_padded
        #rv2 = a2.ravel()
        rv2 = tf.reshape(a2,[-1])
        d, f, g = rv2[0], rv2[1], rv2[2]

        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 (eqn. 15) from [2]
        b /= 2.
        d /= 2.
        f /= 2.

        #tf.print("xparams",self.xparams)

        # finding center of ellipse [eqn.19 and 20] from [2]
        x0 = (c * d - b * f) / (b ** 2. - a * c)
        y0 = (a * f - b * d) / (b ** 2. - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from [2]
        numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 \
                    - 2 * b * d * f - a * c * g
        term = tf.math.sqrt((a - c) ** 2 + 4 * b ** 2)
        denominator1 = (b ** 2 - a * c) * (term - (a + c))
        denominator2 = (b ** 2 - a * c) * (- term - (a + c))
        width = tf.math.sqrt(2 * numerator / denominator1)
        height = tf.math.sqrt(2 * numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse
        # to x-axis [eqn. 23] from [2].
        a,b,c = tf.math.real(a), tf.math.real(b), tf.math.real(c)
        phi = 0.5 * tf.math.atan((2. * b) / (a - c))
        if a > c:
            phi += 0.5 * math.pi

        #self.params = tnp.nan_to_num([x0, y0, width, height, phi]).tolist()
        x0, y0, width, height, phi = tf.math.real(x0), tf.math.real(y0), tf.math.real(width), tf.math.real(height), tf.math.real(phi)
        w  = [x0, y0, width, height, phi]
        #tf.print("w", w)

        self.params = tf.where(tf.math.is_nan(w), tf.ones_like(w) * NUMBER, w)

        #self.params = [float(tf.math.real(x)) for x in self.params]
        pms = self.params
        p1, p2, p3, p4, p5 =  tf.math.real(pms[0]), tf.math.real(pms[1]), tf.math.real(pms[2]), tf.math.real(pms[3]), tf.math.real(pms[4])
        ''' '''
        self.params = [p1, p2, p3, p4, p5]
        #self.xparams = (a, b, c, d, f, g)
        #self.params = [1, 1, 1, 1, 1]
        #tf.print("end of fit")
        return True

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the ellipse is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        """

        _check_data_dim(data, dim=2)

        xc, yc, a, b, theta = self.params

        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        def fun(t, xi, yi):
            ct = math.cos(t)
            st = math.sin(t)
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt) ** 2 + (yi - yt) ** 2

        # def Dfun(t, xi, yi):
        #     ct = math.cos(t)
        #     st = math.sin(t)
        #     xt = xc + a * ctheta * ct - b * stheta * st
        #     yt = yc + a * stheta * ct + b * ctheta * st
        #     dfx_t = - 2 * (xi - xt) * (- a * ctheta * st
        #                                - b * stheta * ct)
        #     dfy_t = - 2 * (yi - yt) * (- a * stheta * st
        #                                + b * ctheta * ct)
        #     return [dfx_t + dfy_t]

        residuals = tnp.empty((N, ), dtype=tnp.float64)

        # initial guess for parameter t of closest point on ellipse
        t0 = tnp.arctan2(y - yc, x - xc) - theta

        # determine shortest distance to ellipse for each point
        for i in range(N):
            xi = x[i]
            yi = y[i]
            # faster without Dfun, because of the python overhead
            t, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
            residuals[i] = tnp.sqrt(fun(t, xi, yi))

        return residuals

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """

        if params is None:
            params = self.params

        xc, yc, a, b, theta = params

        ct = tnp.cos(t)
        st = tnp.sin(t)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st

        return tnp.concatenate((x[..., None], y[..., None]), axis=t.ndim)

def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.
    """
    if probability == 0:
        return 0
    if n_inliers == 0:
        return tnp.inf
    inlier_ratio = n_inliers / n_samples
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    return tnp.ceil(tnp.log(nom) / tnp.log(denom))


def ransac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=tnp.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None):
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:

    1. Select `min_samples` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.

    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.

    Parameters
    ----------
    data : [list, tuple of] (N, ...) array
        Data set to which the model is fitted, where N is the number of data
        points and the remaining dimension are depending on model requirements.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:

         * ``success = estimate(*data)``
         * ``residuals(*data)``

        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples : int in range (0, N)
        The minimum number of data points to fit a model to.
    residual_threshold : float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to a high value
        such as 0.99, e is the current fraction of inliers w.r.t. the
        total number of samples, and m is the min_samples value.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is None the `numpy.random.Generator` singleton is
        used.
        If `random_state` is an int, a new ``Generator`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` instance then that
        instance is used.
    initial_inliers : array-like of bool, shape (N,), optional
        Initial samples selection for model estimation


    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N, ) array
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] "RANSAC", Wikipedia, https://en.wikipedia.org/wiki/RANSAC

    Examples
    --------

    Generate ellipse data without tilt and add noise:

    >>> t = tnp.linspace(0, 2 * tnp.pi, 50)
    >>> xc, yc = 20, 30
    >>> a, b = 5, 10
    >>> x = xc + a * tnp.cos(t)
    >>> y = yc + b * tnp.sin(t)
    >>> data = tnp.column_stack([x, y])
    >>> rng = tnp.random.default_rng(203560)  # do not copy this value
    >>> data += rng.normal(size=data.shape)

    Add some faulty data:

    >>> data[0] = (100, 100)
    >>> data[1] = (110, 120)
    >>> data[2] = (120, 130)
    >>> data[3] = (140, 130)

    Estimate ellipse model using all available data:

    >>> model = EllipseModel()
    >>> model.estimate(data)
    True
    >>> tnp.round(model.params)  # doctest: +SKIP
    array([ 72.,  75.,  77.,  14.,   1.])

    Estimate ellipse model using RANSAC:

    >>> ransac_model, inliers = ransac(data, EllipseModel, 20, 3, max_trials=50)
    >>> abs(tnp.round(ransac_model.params))
    array([20., 30., 10.,  6.,  2.])
    >>> inliers  # doctest: +SKIP
    array([False, False, False, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True], dtype=bool)
    >>> sum(inliers) > 40
    True

    RANSAC can be used to robustly estimate a geometric
    transformation. In this section, we also show how to use a
    proportion of the total samples, rather than an absolute number.

    >>> from skimage.transform import SimilarityTransform
    >>> rng = tnp.random.default_rng()
    >>> src = 100 * rng.random((50, 2))
    >>> model0 = SimilarityTransform(scale=0.5, rotation=1,
    ...                              translation=(10, 20))
    >>> dst = model0(src)
    >>> dst[0] = (10000, 10000)
    >>> dst[1] = (-100, 100)
    >>> dst[2] = (50, 50)
    >>> ratio = 0.5  # use half of the samples
    >>> min_samples = int(ratio * len(src))
    >>> model, inliers = ransac((src, dst), SimilarityTransform, min_samples,
    ...                         10,
    ...                         initial_inliers=tnp.ones(len(src), dtype=bool))
    >>> inliers  # doctest: +SKIP
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True])

    """

    best_inlier_num = 0
    best_inlier_residuals_sum = tnp.inf
    best_inliers = []
    validate_model = is_model_valid is not None
    validate_data = is_data_valid is not None

    random_state = tnp.random.default_rng(random_state)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])

    if not (0 < min_samples < num_samples):
        raise ValueError(f"`min_samples` must be in range (0, {num_samples})")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError(
            f"RANSAC received a vector of initial inliers (length "
            f"{len(initial_inliers)}) that didn't match the number of "
            f"samples ({num_samples}). The vector of initial inliers should "
            f"have the same length as the number of samples and contain only "
            f"True (this sample is an initial inlier) and False (this one "
            f"isn't) values.")

    # for the first run use initial guess of inliers
    spl_idxs = (initial_inliers if initial_inliers is not None
                else random_state.choice(num_samples, min_samples,
                                         replace=False))

    # estimate model for current random sample set
    model = model_class()

    num_trials = 0
    # max_trials can be updated inside the loop, so this cannot be a for-loop
    while num_trials < max_trials:
        num_trials += 1

        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]

        # for next iteration choose random sample set and be sure that
        # no samples repeat
        spl_idxs = random_state.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if validate_data and not is_data_valid(*samples):
            continue

        success = model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if validate_model and not is_model_valid(model, *samples):
            continue

        residuals = tnp.abs(model.residuals(*data))
        # consensus set / inliers
        inliers = residuals < residual_threshold
        residuals_sum = residuals.dot(residuals)

        # choose as new best model if number of inliers is maximal
        inliers_count = tnp.count_nonzero(inliers)
        if (
            # more inliers
            inliers_count > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (inliers_count == best_inlier_num
                and residuals_sum < best_inlier_residuals_sum)):
            best_inlier_num = inliers_count
            best_inlier_residuals_sum = residuals_sum
            best_inliers = inliers
            max_trials = min(max_trials,
                             _dynamic_max_trials(best_inlier_num,
                                                 num_samples,
                                                 min_samples,
                                                 stop_probability))
            if (best_inlier_num >= stop_sample_num
                    or best_inlier_residuals_sum <= stop_residuals_sum):
                break

    # estimate final model using all inliers
    if any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)
        if validate_model and not is_model_valid(model, *data_inliers):
            warn("Estimated model is not valid. Try increasing max_trials.")
    else:
        model = None
        best_inliers = None
        warn("No inliers found. Model not fitted")

    return model, best_inliers
