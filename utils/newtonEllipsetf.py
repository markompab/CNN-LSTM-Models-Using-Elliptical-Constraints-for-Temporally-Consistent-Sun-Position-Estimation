import math

import tensorflow as tf


class NeutonEllipse:
    def __init__(self, x, y, width, height, angle=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.errorV = None
        self.t = None
        # self.error=tf.Variable(0.0)
        # tf.print("width: ", width,"height: ", height, "angle: ", angle)

    def rotation_matrix(self):
        """
        Returns the rotation matrix for the ellipse's rotation.
        """
        a = tf.math.cos(self.angle)
        b = tf.math.sin(self.angle)

        h1 = tf.stack([a, -b])
        h2 = tf.stack([b, a])
        rot_matrix = tf.stack([h1, h2], axis=0)

        # return tf.constant([[a, -b], [b, a]])
        # rot_matrix = [[a, -b], [b, a]]
        return rot_matrix

    def get_point(self, angle):
        """
        Returns the point on the ellipse at the specified local angle.
        """
        r = self.rotation_matrix()
        xe = 0.5 * self.width * tf.math.cos(angle)
        ye = 0.5 * self.height * tf.math.sin(angle)
        return tf.tensordot(r, [xe, ye]) + [self.x, self.y]

    def get_points(self, count):
        """
        Returns an array of points around the ellipse in the specified count.
        """
        t = tf.linspace(0, 2 * tf.math.pi, count)
        xe = 0.5 * self.width * tf.math.cos(t)
        ye = 0.5 * self.height * tf.math.sin(t)
        r = self.rotation_matrix()
        # return tf.tensordot(np.column_stack([xe, ye]), r.T) + [self.x, self.y]
        return tf.tensordot(tf.stack([xe, ye], axis=1), r.T) + [self.x, self.y]

    @tf.function
    def find_distance1(self, x, tolerance=1e-8, max_iterations=10000, learning_rate=0.01):
        """
        Finds the minimum distance between the specified point and the ellipse
        using gradient descent.
        """
        x = tf.constant(x)
        r = self.rotation_matrix()
        x2 = tf.tensordot(r.T, x - [self.x, self.y])
        t = tf.math.atan2(x2[1], x2[0])
        a = 0.5 * self.width
        b = 0.5 * self.height
        iterations = 0
        error1 = tf.Variable(tolerance)
        errors = []
        ts = []

        while error1 >= tolerance and iterations < max_iterations:
            cost = tf.math.cos(t)
            sint = tf.math.sin(t)
            x1 = tf.constant([a * cost, b * sint])
            xp = tf.constant([-a * sint, b * cost])
            dp = 2 * tf.tensordot(xp, x1 - x2)
            t -= dp * learning_rate
            error1.assign(tf.math.abs(dp))
            # errors.append(error)
            # ts.append(t)
            iterations += 1

        ts = tf.constant(ts)
        errors = tf.constant(errors)
        y = tf.norm(x1 - x2)
        success = error < tolerance and iterations < max_iterations
        return dict(x=t, y=y, error=error, iterations=iterations, success=success, xs=ts, errors=errors)

    # @tf.function
    def initT(self, val):
        self.t = tf.Variable(val, dtype=tf.float64)
        #var = tf.Variable(val, dtype=type)

    @tf.function
    def find_distance2(self, x, tolerance=1e-8, max_iterations=1000):
        """
        Finds the minimum distance between the specified point and the ellipse
        using Newton's method.
        """

        # x = tf.constant(x, dtype=tf.float64)
        r = self.rotation_matrix()
        '''
        x = tf.cast(x, tf.float64)
        r = tf.cast(r, tf.float64)
        self.x = tf.cast(self.x, tf.float64)    
        self.y = tf.cast(self.y, tf.float64)
        '''

        if (self.t is None):
            self.t = tf.Variable(0., dtype=tf.float64)

        if (self.errorV is None):
            self.errorV = tf.Variable(0., dtype=tf.float64)


        # x2 = tf.tensordot(r.T, x - [self.x, self.y], 1)
        x2 = tf.tensordot(tf.transpose(r), x - [self.x, self.y], 1)
        # x1 = x2#???
        self.t.assign(tf.math.atan2(x2[1], x2[0]))
        # t=tf.math.atan2(x2[1], x2[0])
        a = 0.5 * self.width
        b = 0.5 * self.height

        # If point is inside ellipse, generate better initial angle based on vertices
        if (x2[0] / a) ** 2 + (x2[1] / b) ** 2 < 1:
            # ts = tf.linspace(0, 2 * math.pi, 24, endpoint=False)

            ts = tf.linspace(0., 2 * math.pi, tf.cast(24, tf.int32))
            # ts = tf.linspace(0., 2 * math.pi, tf.cast(24, tf.float64))
            ts = tf.cast(ts, tf.float64)

            xe = a * tf.math.cos(ts)
            ye = b * tf.math.sin(ts)
            # delta = x2 - np.column_stack([xe, ye])
            delta = x2 - tf.stack([xe, ye], axis=1)
            # t = ts[np.argmin(tf.norm(delta, axis=1))]

            self.t.assign(ts[tf.math.argmin(tf.norm(delta, axis=1))])

        iterations = 0
        # error1 = tf.Variable(tolerance)

        self.errorV.assign(tolerance)
        errors = []  # tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        ts = []  # tf.TensorArray(tf.float64, size=0, dynamic_size=True)

        while self.errorV >= tf.constant(tolerance, dtype=tf.float64) and iterations < max_iterations:
            cost = tf.math.cos(self.t)
            sint = tf.math.sin(self.t)

            x1 = tf.stack([a * cost, b * sint])
            xp = tf.stack([-a * sint, b * cost])
            xpp = tf.stack([-a * cost, -b * sint])
            delta = x1 - x2

            # tf.print("xp",xp,"xpp",xpp,"delta",delta)
            dp = tf.tensordot(xp, delta, axes=1)
            dpp = tf.tensordot(xpp, delta, axes=1) + tf.tensordot(xp, xp, axes=1)

            # t -= dp / dpp

            tm = self.t - dp / dpp

            self.t.assign(tm)
            # error = tf.abs(dp / dpp)
            er = tf.math.abs(dp / dpp)

            # tf.print("errorV",self.errorV)
            if (self.errorV is not None):
                self.errorV.assign(er)
            ''' 

            #tf.print("t:",t,"dpp:",dpp)

            errors.append(error)
            #errors =  tf.concat([errors, [error]], 0)
            ts.append(tm)
            #ts =  tf.concat([ts, [tm]], 0)
            '''
            iterations += 1
            # tf.print("iterations:", iterations, "error:", er)

        # f.print("error:", error)

        # ts = tf.constant(ts)
        # tf.print("ts", ts)
        # errors = tf.constant(errors)
        # tf.print("errors", errors)
        y = tf.norm(x1 - x2)
        success = self.errorV < tolerance and iterations < max_iterations

        ''' '''

        if (self.errorV is None):
            self.errorV = tf.Variable(0., dtype=tf.float64)
            #self.errorV.assign(0)
        # tf.print("t:",self.t,"errorV:",self.errorV)
        # t,y,error,iterations,success,ts,errors = 1,1,1,1,1,1,1

        # tf.cond(tf.math.is_nan(self.errorV), self.initV(self.errorV, 0., tf.float64), lambda: tf.print(""))


        return dict(x=self.t, y=y, error=self.errorV, iterations=iterations, success=success, xs=ts, errors=errors)
        #return self.errorV