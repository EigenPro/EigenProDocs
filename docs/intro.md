## Introduction
The EigenPro2 is proposed to achieve very fast, scalable, and accurate training for kernel machines.
A detailed description of the method can be found in paper -
["Learning kernels that adapt to GPU"](https://arxiv.org/abs/1806.06144).


## Experimental results

### Comparison to ThunderSVM
These experiments were done by running ThunderSVM on several different datasets until it converged, then running EigenPro until it achieved the same validation error or less. (We have found that running it for additional epochs can decrease the error much further). In these experiments, EigenPro is more than 10x faster than ThunderSVM (which is itself about 100x faster than libSVM). Additionally, EigenPro requires much less memory, and can be run even on personal laptop.

<table>
  <tr>
    <td>Dataset</td>
    <td colspan="2">ThunderSVM</td>
    <td colspan="2">EigenPro</td>
  </tr>
  <tr>
    <td></td>
    <td>Time</td>
    <td>Test Error</td>
    <td>Time</td>
    <td>Test Error</td>
  </tr>
  <tr>
    <td>MNIST 60K</td>
    <td>30.7 sec</td>
    <td>1.48%</td>
    <td>3.04 sec</td>
    <td>1.45%</td>
  </tr>
  <tr>
    <td>CIFAR 50K</td>
    <td>70.2 sec</td>
    <td>53.67%</td>
    <td>2.86 sec</td>
    <td>52.78%</td>
  </tr>
  <tr>
    <td>SVHN 73K</td>
    <td>141.5 sec</td>
    <td>19.33%</td>
    <td>12.14 sec</td>
    <td>19.21%</td>
  </tr>
</table>

### Classification Error (MNIST)
In these experiments, EigenPro (Primal) achieves classification error 1.23%, after only 10 epochs. In comparison, Pegasos takes 160 epochs to reach the same error. Although the number of random features used by EigenPro (Random) and RF/DSGD is 6 * 10^4, same as the number of training points, methods using random features deliver slighly worse performance. Specifically, RF/DSGD has error rate 1.71% after 40 epochs and Pegasos reaches error rate 1.65% after the same number of epochs.

<table>
  <tr>
    <th rowspan="2">#Epochs</th>
    <th colspan="4">Primal</th>
    <th colspan="4">Random Fourier Feature</th>
  </tr>
  <tr>
    <td align="center" colspan="2">EigenPro</td>
    <td align="center" colspan="2">Pegasos</td>
    <td align="center" colspan="2">EigenPro</td>
    <td align="center" colspan="2">RF/DSGD</td>
  </tr>
  <tr>
    <td></td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0.92%</td>
    <td>2.03%</td>
    <td>5.12%</td>
    <td>5.21%</td>
    <td>0.80%</td>
    <td>1.93%</td>
    <td>5.21%</td>
    <td>5.33%</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.10%</td>
    <td>1.44%</td>
    <td>2.36%</td>
    <td>2.84%</td>
    <td>0.12%</td>
    <td>1.49%</td>
    <td>2.48%</td>
    <td>2.98%</td>
  </tr>
  <tr>
    <td>10</td>
    <td>0.01%</td>
    <td>1.23%</td>
    <td>1.58%</td>
    <td>2.32%</td>
    <td>0.03%</td>
    <td><b>1.44%</b></td>
    <td>1.66%</td>
    <td>2.37%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>0.0%</td>
    <td><b>1.20%</b></td>
    <td>0.90%</td>
    <td>1.93%</td>
    <td>0.01%</td>
    <td>1.45%</td>
    <td>0.98%</td>
    <td>2.03%</td>
  </tr>
  <tr>
    <td>40</td>
    <td>0.0%</td>
    <td>1.20%</td>
    <td>0.39%</td>
    <td>1.65%</td>
    <td>0.0%</td>
    <td>1.46%</td>
    <td>0.49%</td>
    <td>1.71%</td>
  </tr>
</table>


### Training Time per Epoch

<table>
  <tr>
    <th rowspan="2">Computing<br>Resource</th>
    <th colspan="2">Primal</th>
    <th colspan="2">Random Fourier Feature</th>
  </tr>
  <tr>
    <td align="center">EigenPro</td>
    <td align="center">Pegasos</td>
    <td align="center">EigenPro</td>
    <td align="center">RF/DSGD</td>
  </tr>
  <tr>
    <td>One GTX Titan X (Maxwell)</td>
    <td align="center">4.8s</td>
    <td align="center">4.6s</td>
    <td align="center">2.2s</td>
    <td align="center">2.0s</td>
  </tr>
  <tr>
    <td>One GTX Titan Xp (Pascal)</td>
    <td align="center">2.6s</td>
    <td align="center">2.3s</td>
    <td align="center">1.1s</td>
    <td align="center">1.0s</td>
  </tr>
  <tr>
    <td>Two Xeon E5-2620</td>
    <td align="center">72s</td>
    <td align="center">70s</td>
    <td align="center">78s</td>
    <td align="center">72s</td>
  </tr>
</table>

### EigenPro Preprocessing Time
In our experiments we construct the EigenPro preconditioner by computing the top 160 approximate eigenvectors for a subsample matrix with 4800 points using Randomized SVD (RSVD).

<table>
  <tr>
    <th>Computing<br>Resource</th>
    <th>RSVD Time<br>(k = 160, m = 4800)</th>
  </tr>
  <tr>
    <td>One GTX Titan X (Maxwell)</td>
    <td align="center">7.6s</td>
  </tr>
  <tr>
    <td>One GTX Titan Xp (Pascal)</td>
    <td align="center">6.3s</td>
  </tr>
  <tr>
    <td>Two Xeon E5-2620</td>
    <td align="center">17.7s</td>
  </tr>
</table>
