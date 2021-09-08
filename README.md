# Numerical Inversion of Injective Immersion

This repository contains simulations attempting to use gradient descent and Newton algorithms to solve problems of the type ys = T(xs), knowing ys, where T is an injective immersion.

The examples are reported and explained in *Remarks about the numerical inversion of injective nonlinear maps* by V. Andrieu and P. Bernard, which will be presented at the 2021 IEEE Conference on Decision and Control.

Two examples are presented :

- in main_manipulator, T describes the 3d spatial position of a three-link manipulator given its angular positions : neither the Newton nor gradient method work in this case.

- in main_global, T is a simple 2d global diffeomorphism, but which is not surjective and whose image set is not convex :  neither the Newton nor gradient method work in this case either, but we show that by extending the image of T by composition with a surjective diffeomophism, those algorithms work and become global.
