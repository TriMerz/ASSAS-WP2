========== !! DATA INFO !! ==========
1. ASTEC itself creates the CESAR_IO family from the first macro time-step, i.e. the first saved database
   is not read by the script.
2. Indeed the first CESAR time-step inside the erster macro time-step is made of the STEDAY-STATE last values, included
   the STEADY-STATE values STEPBEG = 0 and STEPEND = 10.
3. Using that values, ASTEC is not able to reac the convergence, resulting in CONV = 0.
4. Therefore data has been cleaned by the non converged value and only CONV = 1 are stored.
5. The number of saves and dtmacro are saved every MACRO time-step while: dtcesar and varprim vector are saved every
   CESAR time-step.
6. The matrix of varprim values is used to calculate the incremental/decremental value of every features
   between each CESAR time-step.
7. Since we are interested in the values variation between two consecutive time steps, the first TRANSIENT, CONVERGED
   step has been removed from the matrix of differential values, resulting in [n-1] rows.
