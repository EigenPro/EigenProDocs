# How it works

EigenPro accelerates the convergence of SGD iteration when minimizing linear and kernel least squares, defined as

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\arg&space;\min_{{\pmb&space;\alpha}&space;\in&space;\mathcal{H}}&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;{&space;(\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;-&space;y_i)^2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\arg&space;\min_{{\pmb&space;\alpha}&space;\in&space;\mathcal{H}}&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;{&space;(\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;-&space;y_i)^2}" title="\arg \min_{{\pmb \alpha} \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^{n} { (\left \langle {\pmb \alpha}, {\pmb x}_i \right \rangle_\mathcal{H} - y_i)^2}" /></a>
</p>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{({\pmb&space;x}_i,&space;y_i)\}_{i=1}^n" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\{({\pmb&space;x}_i,&space;y_i)\}_{i=1}^n" title="\{({\pmb x}_i, y_i)\}_{i=1}^n" /></a>
is the labeled training data. Let
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X&space;\doteq&space;({\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n)^T,&space;{\pmb&space;y}&space;\doteq&space;(y_1,&space;\ldots,&space;y_n)^T" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;X&space;\doteq&space;({\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n)^T,&space;{\pmb&space;y}&space;\doteq&space;(y_1,&space;\ldots,&space;y_n)^T" title="X \doteq ({\pmb x}_1, \ldots, {\pmb x}_n)^T, {\pmb y} \doteq (y_1, \ldots, y_n)^T" /></a>
.


Consider the linear setting where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{H}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathcal{H}" title="\mathcal{H}" /></a>
is a vector space and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;\doteq&space;{\pmb&space;\alpha}^T&space;{\pmb&space;x}_i" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;\doteq&space;{\pmb&space;\alpha}^T&space;{\pmb&space;x}_i" title="\left \langle {\pmb \alpha}, {\pmb x}_i \right \rangle_\mathcal{H} \doteq {\pmb \alpha}^T {\pmb x}_i" /></a>
. The corresponding standard gradient descent iteration is hence,

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex={\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;\eta&space;(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" target="_blank"><img src="https://latex.codecogs.com/png.latex?{\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;\eta&space;(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" title="{\pmb \alpha} \leftarrow {\pmb \alpha} - \eta (H {\pmb \alpha} - {\pmb b})" /></a>
</p>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;H&space;\doteq&space;X^T&space;X" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;H&space;\doteq&space;X^T&space;X" title="H \doteq X^T X" /></a>
is the covariance matrix and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\pmb&space;b}&space;\doteq&space;X^T{\pmb&space;y}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;{\pmb&space;b}&space;\doteq&space;X^T{\pmb&space;y}" title="{\pmb b} \doteq X^T{\pmb y}" /></a>
. The step size is automatically set as
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\eta&space;\leftarrow&space;1.5&space;\cdot&space;\lambda_1(H)^{-1}" target="_blank"><img align="center" src="https://latex.codecogs.com/gif.latex?\inline&space;\eta&space;\leftarrow&space;1.5&space;\cdot&space;\lambda_1(H)^{-1}" title="\eta \leftarrow 1.5 \cdot \lambda_1(H)^{-1}" /></a>
to ensure fast convergence. Note that the top eigenvalue of the covariance is calculated approximately.
We then construct EigenPro preconditioner P using the approximate top eigensystem of H,
which can be efficiently calculated when H has fast eigendecay.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;\doteq&space;I&space;-&space;\sum_{i=1}^k&space;{(1&space;-&space;\tau&space;\frac{\lambda_{k&plus;1}(H)}&space;{\lambda_i(H)})&space;{\pmb&space;e}_i(H)&space;{\pmb&space;e}_i(H)^T}" target="_blank"><img src="https://latex.codecogs.com/png.latex?P&space;\doteq&space;I&space;-&space;\sum_{i=1}^k&space;{(1&space;-&space;\tau&space;\frac{\lambda_{k&plus;1}(H)}&space;{\lambda_i(H)})&space;{\pmb&space;e}_i(H)&space;{\pmb&space;e}_i(H)^T}" title="P \doteq I - \sum_{i=1}^k {(1 - \tau \frac{\lambda_{k+1}(H)} {\lambda_i(H)}) {\pmb e}_i(H) {\pmb e}_i(H)^T}" /></a>
</p>

Here we select
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tau&space;\leq&space;1" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\tau&space;\leq&space;1" title="\tau \leq 1" /></a>
to counter the negative impact of eigensystem approximation error on convergence.
The EigenPro iteration then runs as follows,

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex={\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;(\eta&space;\frac{\lambda_1(H)}{\lambda_{k&plus;1}(H)})&space;P(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" target="_blank"><img src="https://latex.codecogs.com/png.latex?{\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;(\eta&space;\frac{\lambda_1(H)}{\lambda_{k&plus;1}(H)})&space;P(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" title="{\pmb \alpha} \leftarrow {\pmb \alpha} - (\eta \frac{\lambda_1(H)}{\lambda_{k+1}(H)}) P(H {\pmb \alpha} - {\pmb b})" /></a>
</p>

With larger
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\lambda_1(H)&space;/&space;\lambda_{k&plus;1}(H)" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\lambda_1(H)&space;/&space;\lambda_{k&plus;1}(H)" title="\lambda_1(H) / \lambda_{k+1}(H)" /></a>
, EigenPro iteration yields higher convergence acceleration over standard (stochastic) gradient descent.
This is especially critical in the kernel setting where (widely used) smooth kernels have exponential eigendecay.
Note that in such setting
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{H}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathcal{H}" title="\mathcal{H}" /></a>
is typically an RKHS (reproducing kernel Hilbert space) of infinite dimension. Thus it is necessary to parametrize the (approximate) solution in a subspace of finite dimension (e.g.
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathrm{span}_{{\pmb&space;x}&space;\in&space;\{&space;{\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n&space;\}}&space;\{&space;k(\cdot,&space;{\pmb&space;x})&space;\}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathrm{span}_{{\pmb&space;x}&space;\in&space;\{&space;{\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n&space;\}}&space;\{&space;k(\cdot,&space;{\pmb&space;x})&space;\}" title="\mathrm{span}_{{\pmb x} \in \{ {\pmb x}_1, \ldots, {\pmb x}_n \}} \{ k(\cdot, {\pmb x}) \}" /></a>
).
See [the paper](https://arxiv.org/abs/1703.10622) for more details on the kernel setting and some theoretical results.
