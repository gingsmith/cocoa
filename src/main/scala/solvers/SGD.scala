package distopt.solvers

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import distopt.utils._
import breeze.linalg.{Vector, NumericOps, DenseVector, SparseVector}


object SGD {

  /**
   * Implementation of distributed SGD, for mini-batch as well as localSGD variants.
   * Using hinge-loss SVM objective.
   * 
   * @param data RDD of all data examples
   * @param params Algorithmic parameters
   * @param debug Systems/debugging parameters
   * @param local Whether to perform localSGD (local=true) or classical mini-batch SGD (local=false)
   * @return
   */
  def runSGD(
    data: RDD[LabeledPoint],
    params: Params,
    debug: DebugParams,
    local: Boolean) : Vector[Double] = {
    
    var dataArr = data.mapPartitions(x => Iterator(x.toArray))
    val parts = data.partitions.size 	// number of partitions of the data, K in the paper
    println("\nRunning SGD (with local updates = "+local+") on "+params.n+" data examples, distributed over "+parts+" workers")
    
    // initialize w
    var w = params.wInit.copy
    
    var scaling = 1.0
    if (local) {
      scaling = params.beta / parts
    } else {
      scaling = params.beta / (parts * params.localIters)
    }

    for(t <- 1 to params.numRounds){

      // update step size
      val step = 1 / (params.lambda * (t))

      if (!local) {
        // scale weight vector
        val scale = 1.0 - (step * params.lambda)
        w :*= scale
      }

      // find updates to w
      val updates = dataArr.mapPartitions(partitionUpdate(_, w, params.lambda, ((t-1) * params.localIters * parts), params.localIters, local, parts, debug.seed + t), preservesPartitioning = true).persist()
      val primalUpdates = updates.reduce(_ + _)
      if (local) {
        w += (primalUpdates * scaling)
      } else {
        w += (primalUpdates * (step * scaling))
      }

      // optionally calculate errors
      if (debug.debugIter > 0 && t % debug.debugIter == 0) {
        println("Iteration: " + t)
        println("primal objective: " + OptUtils.computePrimalObjective(data, w, params.lambda))
        if (debug.testData != null) { println("test error: " + OptUtils.computeClassificationError(debug.testData, w)) }
      }
    }

    return w
  }


  /**
   * Performs one round of local updates using SGD steps on the local points, 
   * Will perform localIters many updates per worker.
   * 
   * @param localData
   * @param wInit
   * @param lambda
   * @param t
   * @param localIters
   * @param local
   * @param parts
   * @param seed
   * @return
   */
  def partitionUpdate(
    localData: Iterator[Array[LabeledPoint]], 
    wInit: Vector[Double], 
    lambda:Double, 
    t:Double, 
    localIters:Int, 
    local:Boolean, 
    parts:Int,
    seed: Int) : Iterator[Vector[Double]] = {

    val dataArr = localData.next()
    val nLocal = dataArr.length
    var r = new scala.util.Random(seed)
    var w = wInit.copy
    var deltaW = Vector.zeros[Double](wInit.length)

    // perform updates
    for (i <- 1 to localIters) {

      val step = 1 / (lambda * (t + i))

      // randomly select an element
      val idx = r.nextInt(nLocal)
      val currPt = dataArr(idx)
      var y = currPt.label
      val x = currPt.features

      // calculate stochastic sub-gradient (here for SVM hinge loss)
      val eval = 1.0 - (y * (x.dot(w)))

      if (local) {
        // scale weight vector
        val scale = 1.0 - (step * lambda)
        w :*= scale
      }

      // stochastic sub-gradient, update
      if (eval > 0) {
        val update = x * y
        deltaW += update
        if (local){
          w += (update * step)
        }
      }

      if (local) {
        deltaW = w - wInit
      }
    }

    // return change in weight vector
    return Iterator(deltaW)
  }

}
