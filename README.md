# CoCoA - A Framework for Communication-Efficient Distributed Optimization

New! We've added support for faster additive udpates with CoCoA+. See more information [here](http://arxiv.org/abs/1502.03508).

This code performs a comparison of 5 distributed algorithms for training of machine learning models, using [Apache Spark](http://spark.apache.org). The implemented algorithms are
 - _CoCoA+_
 - _CoCoA_
 - mini-batch stochastic dual coordinate ascent (_mini-batch SDCA_)
 - stochastic subgradient descent with local updates (_local SGD_)
 - mini-batch stochastic subgradient descent (_mini-batch SGD_)

The present code trains a standard SVM (hinge-loss, l2-regularized) using SDCA as a local solver, and reports training and test error, as well as the duality gap certificate if the method is primal-dual. The code can be easily adapted to include other internal solvers or solve other objectives.

## Getting Started
How to run the code locally:

```
sbt/sbt assembly
./run-demo-local.sh
```

(For the `sbt` script to run, make sure you have downloaded CoCoA into a directory whose path contains no spaces.)

## References
The CoCoA+ and CoCoA algorithmic frameworks are described in more detail in the following papers:

_Ma, C., Smith, V., Jaggi, M., Jordan, Michael I., Richtarik, P., & Takac, M. [Adding vs. Averaging in Distributed Primal-Dual Optimization](http://arxiv.org/abs/1502.03508). ICML 2015 - International Conference on Machine Learning._

_Jaggi, M., Smith, V., Takac, M., Terhorst, J., Krishnan, S., Hofmann, T., & Jordan, M. I. [Communication-Efficient Distributed Dual Coordinate Ascent](http://papers.nips.cc/paper/5599-communication-efficient-distributed-dual-coordinate-ascent) (pp. 3068–3076). NIPS 2014 - Advances in Neural Information Processing Systems 27._
