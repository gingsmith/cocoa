package distopt.solvers

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import distopt.utils._
import breeze.linalg.{Vector, NumericOps, DenseVector, SparseVector}


object MinibatchCD {

  /**
   * Mini-batch SDCA, that is dual coordinate descent for standard hinge-loss SVM
   * 
   * @param data RDD of all data examples
   * @param params Algorithmic parameters
   * @param debug Systems/debugging parameters
   * @return
   */
  def runMbCD(
    data: RDD[LabeledPoint],
    params: Params,
    debug: DebugParams) : (Vector[Double], RDD[Vector[Double]]) = {
    
    val parts = data.partitions.size 	// number of partitions of the data, K in the paper
    println("\nRunning Mini-batch CD on "+params.n+" data examples, distributed over "+parts+" workers")
    
    // initialize alpha, w
    var alphaVars = data.map(x => 0.0).cache()
    var alpha = alphaVars.mapPartitions(x => Iterator(Vector(x.toArray)))
    var dataArr = data.mapPartitions(x => Iterator(x.toArray))
    var w = params.wInit.copy
    val scaling = params.beta / (parts * params.localIters)

    for(t <- 1 to params.numRounds){

      // zip alpha with data
      val zipData = alpha.zip(dataArr)

      // find updates to alpha, w
      val updates = zipData.mapPartitions(partitionUpdate(_, w, params.localIters, params.lambda, params.n, scaling, debug.seed + t), preservesPartitioning = true).persist()
      alpha = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(_ + _)
      w += (primalUpdates * scaling)

      // optionally calculate errors
      if (debug.debugIter > 0 && t % debug.debugIter == 0) {
        println("Iteration: " + t)
        println("primal objective: " + OptUtils.computePrimalObjective(data, w, params.lambda))
        println("primal-dual gap: " + OptUtils.computeDualityGap(data, w, alpha, params.lambda))
        if (debug.testData != null) { println("test error: " + OptUtils.computeClassificationError(debug.testData, w)) }
      }

      // optionally checkpoint RDDs
      if(t % debug.chkptIter == 0){
        zipData.checkpoint()
        alpha.checkpoint()
      }
    }

    return (w, alpha)
  }


  /**
   * Performs one round of mini-batch CD updates
   *
   * @param zipData
   * @param winit
   * @param localIters
   * @param lambda
   * @param n
   * @param scaling Possible scaling beta/(K*H) in the paper
   * @param seed
   * @return
   */
  private def partitionUpdate(
    zipData: Iterator[(Vector[Double], Array[LabeledPoint])],//((Int, Double), SparseClassificationPoint)],
    wInit: Vector[Double], 
    localIters: Int, 
    lambda: Double, 
    n: Int, 
    scaling: Double,
    seed: Int): Iterator[(Vector[Double], Vector[Double])] = {

    val zipPair = zipData.next()
    var localData = zipPair._2
    var alpha = zipPair._1
    val alphaOld = alpha.copy
    var w = wInit
    val nLocal = localData.length
    var r = new scala.util.Random(seed)
    var deltaW = DenseVector.zeros[Double](wInit.length)

    // perform local udpates
    for (i <- 1 to localIters) {

      // randomly select a local example
      val idx = r.nextInt(nLocal)
      val currPt = localData(idx)
      var y = currPt.label
      val x = currPt.features

      // compute hinge loss gradient
      val grad = (y * (x.dot(w)) - 1.0) * (lambda * n)

      // compute projected gradient
      var proj_grad = grad
      if (alpha(idx) <= 0.0)
        proj_grad = Math.min(grad, 0)
      else if (alpha(idx) >= 1.0)
        proj_grad = Math.max(grad, 0)

      if (Math.abs(proj_grad) != 0.0 ) {
        val qii  = Math.pow(x.norm(2), 2)
        var newAlpha = 1.0
        if (qii != 0.0) {
          newAlpha = Math.min(Math.max((alpha(idx) - (grad / qii)), 0.0), 1.0)
        }

        // update primal and dual variables
        val update = x * (y * (newAlpha - alpha(idx)) / (lambda * n))
        deltaW += update
        alpha(idx) = newAlpha
      }
    }

    val deltaAlpha = alpha - alphaOld   
    alpha = alphaOld + (deltaAlpha * scaling)

    // return weight vector, alphas
    return Iterator((deltaW, alpha))
  }

}
