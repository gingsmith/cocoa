package distopt.solvers

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import distopt.utils.Implicits._
import distopt.utils._

object SGD {

  /**
   * Implementation of distributed SGD, for mini-batch as well as localSGD variants.
   * Using hinge-loss SVM objective.
   * 
   * @param sc
   * @param data RDD of all data examples
   * @param wInit initial weight vector (has to be zero)
   * @param numRounds number of outer iterations T in the paper
   * @param localIters number of inner localSDCA iterations, H in the paper
   * @param lambda the regularization parameter
   * @param local whether to perform localSGD (local=true) or classical mini-batch SGD (local=false)
   * @param beta scaling parameter. beta=1 gives averaging, beta=K=data.partitions.size gives (aggressive) adding
   * @param chkptIter checkpointing the resulting RDDs from time to time, to ensure persistence and shorter dependencies
   * @param testData
   * @param debugIter
   * @return
   */
  def runSGD(
    sc: SparkContext, 
    data: RDD[SparseClassificationPoint],
    n: Int,
    wInit: Array[Double], 
    numRounds: Int, 
    localIters: Int, 
    lambda: Double,
    local: Boolean, 
    beta: Double, 
    chkptIter: Int,
    testData: RDD[SparseClassificationPoint],
    debugIter: Int) : Array[Double] = {
    
    val parts = data.partitions.size 	// number of partitions of the data, K in the paper
    println("\nRunning SGD (with local updates = "+local+") on "+n+" data examples, distributed over "+parts+" workers")
    
    // initialize w
    var w = wInit
    
    var scaling = 1.0
    if (local) {
      scaling = beta / parts
    } else {
      scaling = beta / (parts * localIters)
    }

    for(t <- 1 until numRounds+1){

      // update step size
      val step = 1/(lambda*(t))

      if (!local) {
        // scale weight vector
        val scale = 1.0-(step*lambda)
        w = w.map(_*scale)
      }

      // find updates to w
      val updates = data.mapPartitions(partitionUpdate(_, w, lambda, ((t-1) * localIters * parts), localIters, local, parts), preservesPartitioning = true).persist()
      val primalUpdates = updates.reduce(_ plus _)
      if (local) {
        w = primalUpdates.times(scaling).plus(w)
      } else {
        w = primalUpdates.times(step * scaling).plus(w)
      }

      // optionally calculate errors
      if (debugIter>0 && t % debugIter == 0) {
        println("Iteration: " + t)
        println("primal objective: " + OptUtils.computePrimalObjective(data, w, lambda))
        if (testData != null) { println("test error: " + OptUtils.computeClassificationError(testData, w)) }
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
   * @return
   */
  def partitionUpdate(
    localData: Iterator[SparseClassificationPoint], 
    wInit: Array[Double], 
    lambda:Double, 
    t:Double, 
    localIters:Int, 
    local:Boolean, 
    parts:Int) : Iterator[Array[Double]] = {

    val dataArr = localData.toArray
    val nLocal = dataArr.length
    var r = new scala.util.Random
    var w = wInit.clone
    var deltaW = Array.fill(wInit.length)(0.0)

    // perform updates
    for (i <- 0 until localIters) {

      val step = 1/(lambda*(t+i+1))

      // randomly select an element
      val idx = r.nextInt(nLocal)
      val currPt = dataArr(idx)
      var y = currPt.label
      val x = currPt.features

      // calculate stochastic sub-gradient (here for SVM hinge loss)
      val eval = 1.0 - (y*(x.dot(w)))

      if (local) {
        // scale weight vector
        val scale = 1.0 - (step * lambda)
        w = w.map(_ * scale)
      }

      // stochastic sub-gradient, update
      if (eval > 0) {
        val update = x.times(y)
        deltaW = update.plus(deltaW)
        if (local){
          w = (update.times(step)).plus(w)
        }
      }

      if (local) {
        deltaW = (wInit.times(-1.0)).plus(w)
      }

    }

    // return change in weight vector
    return Iterator(deltaW)
  }

}
