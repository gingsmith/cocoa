# CoCoA - Communication-Efficient Distributed Coordinate Ascent

The demo code performs a comparison of 4 distributed algorithms for training of machine learning models, using [Apache Spark](http://mspark.apache.org/). The implemented algorithms are
 - CoCoA
 - mini-batch stochastic dual coordinate ascent (mini-batch SDCA)
 - stochastic subgradient descent with local updates (local SGD)
 - mini-batch stochastic subgradient descent (mini-batch SGD)
The present code trains a standard SVM (hinge-loss, l2-regularized), and reports training as well as test error, as well as the duality gap certificate if the method is primal-dual.

## Getting Started
How to run the code locally:

```
sbt/sbt assembly
./run-demo-local.sh
```

(For the `sbt` script to run, make sure you have downloaded cocoa into a directory whose path contains no spaces.)

## References
The CoCoA algorithm framework is described in more details in the following paper:

_Jaggi, M., Smith, V., Takac, M., Terhorst, J., Krishnan, S., Hofmann, T., & Jordan, M. I. (2014) [Communication-Efficient Distributed Dual Coordinate Ascent](http://papers.nips.cc/paper/5599-communication-efficient-distributed-dual-coordinate-ascent) (pp. 3068–3076). NIPS 2014 - Advances in Neural Information Processing Systems 27._
