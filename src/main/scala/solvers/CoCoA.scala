package distopt.solvers

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import distopt.utils.Implicits._
import distopt.utils._

object CoCoA {

  /**
   * CoCoA - Communication-efficient distributed dual Coordinate Ascent.
   * Using LocalSDCA as the local dual method. Here implemented for standard 
   * hinge-loss SVM. For other objectives, adjust localSDCA accordingly.
   * 
   * @param sc
   * @param data RDD of all data examples
   * @param wInit initial weight vector (has to be zero)
   * @param numRounds number of outer iterations T in the paper
   * @param localIters number of inner localSDCA iterations, H in the paper
   * @param lambda the regularization parameter
   * @param beta scaling parameter. beta=1 gives averaging, beta=K=data.partitions.size gives (aggressive) adding
   * @param chkptIter checkpointing the resulting RDDs from time to time, to ensure persistence and shorter dependencies
   * @param testData
   * @param debugIter
   * @return
   */
  def runCoCoA(
    sc: SparkContext, 
    data: RDD[SparseClassificationPoint],
    n: Int,
    wInit: Array[Double], 
    numRounds: Int, 
    localIters: Int, 
    lambda: Double, 
    beta: Double, 
    chkptIter: Int, 
    testData: RDD[SparseClassificationPoint], 
    debugIter: Int,
    seed: Int) : (Array[Double], RDD[(Int, Double)]) = {
    
    val parts = data.partitions.size 	// number of partitions of the data, K in the paper
    println("\nRunning CoCoA on "+n+" data examples, distributed over "+parts+" workers")
    
    // initialize alpha, w
    var alpha = data.map(x => (x.index, 0.0)).cache()
    var w = wInit
    val scaling = beta / parts;

    for(t <- 1 to numRounds){

      // zip alpha with data
      val zipData = alpha.zip(data)

      // find updates to alpha, w
      val updates = zipData.mapPartitions(partitionUpdate(_,w,localIters,lambda,n,scaling,seed+t),preservesPartitioning=true).persist()
      alpha = updates.map(kv => kv._2)
      val primalVariables = updates.map(kv => kv._1)
      val primalUpdates = primalVariables.mapPartitions(x => Iterator(x.next())).reduce(_ plus _)
      w = primalUpdates.times(scaling).plus(w)

      // optionally calculate errors
      if (debugIter>0 && t % debugIter == 0) {
        println("Iteration: " + t)
        println("primal objective: " + OptUtils.computePrimalObjective(data, w, lambda))
        println("primal-dual gap: " + OptUtils.computeDualityGap(data, w, alpha, lambda))
        if (testData != null) { println("test error: " + OptUtils.computeClassificationError(testData, w)) }
      }

      // optionally checkpoint RDDs
      if(t % chkptIter == 0){
        zipData.checkpoint()
        alpha.checkpoint()
      }
    }

    return (w, alpha)
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   *
   * @param zipData
   * @param winit
   * @param localIters
   * @param lambda
   * @param n
   * @param scaling this is the scaling factor beta/K in the paper
   * @param seed
   * @return
   */
  private def partitionUpdate(
    zipData: Iterator[((Int, Double), SparseClassificationPoint)],
    wInit: Array[Double], 
    localIters: Int, 
    lambda: Double, 
    n: Int, 
    scaling: Double,
    seed: Int): Iterator[(Array[Double], (Int, Double))] = {

    val zipArr = zipData.toArray
    var localData = zipArr.map(x => x._2)
    var alpha = zipArr.map(x => x._1._2)
    val indices = (0 to localData.length-1).map(x => localData(x).index).toArray
    val alphaOld = alpha.clone
    val (deltaAlpha, deltaW) = localSDCA(localData, wInit, localIters, lambda, n, alpha, alphaOld, seed)
    
    alpha = alphaOld.plus(deltaAlpha.times(scaling))
    var wArray = Array.fill(localData.length)(Array(0.0))
    wArray(0) = deltaW

    // return weight vector, alphas
    return wArray.zip(indices.zip(alpha)).iterator
  }

  /**
   * This is an implementation of LocalDualMethod, here LocalSDCA (coordinate ascent),
   * with taking the information of the other workers into account, by respecting the
   * shared wInit vector.
   * Here we perform coordinate updates for the SVM dual objective (hinge loss).
   *
   * Note that SDCA for hinge-loss is equivalent to LibLinear, where using the
   * regularization parameter  C = 1.0/(lambda*numExamples), and re-scaling
   * the alpha variables with 1/C.
   *
   * @param localData the local data examples
   * @param wInit
   * @param localIters number of local coordinates to update
   * @param lambda
   * @param n global number of points (needed for the primal-dual correspondence)
   * @param alpha
   * @param alphaOld
   * @param seed
   * @return deltaAlpha and deltaW, summarizing the performed local changes, see paper
   */
  def localSDCA(
    localData: Array[SparseClassificationPoint],
    wInit: Array[Double], 
    localIters: Int, 
    lambda: Double, 
    n: Int,
    alpha: Array[Double], 
    alphaOld: Array[Double],
    seed: Int): (Array[Double], Array[Double]) = {
    var w = wInit
    val nLocal = localData.length
    var r = new scala.util.Random(seed)
    var deltaW = Array.fill(wInit.length)(0.0)

    // perform local udpates
    for (i <- 0 until localIters) {

      // randomly select a local example
      val idx = r.nextInt(nLocal)
      val currPt = localData(idx)
      var y = currPt.label
      val x = currPt.features

      // compute hinge loss gradient
      val grad = (y*(x.dot(w)) - 1.0)*(lambda*n)

      // compute projected gradient
      var proj_grad = grad
      if (alpha(idx) <= 0.0)
        proj_grad = Math.min(grad,0)
      else if (alpha(idx) >= 1.0)
        proj_grad = Math.max(grad,0)

      if (Math.abs(proj_grad) != 0.0 ) {
        val qii  = x.dot(x)
        var newAlpha = 1.0
        if (qii != 0.0) {
          newAlpha = Math.min(Math.max((alpha(idx) - (grad / qii)), 0.0), 1.0)
        }

        // update primal and dual variables
        val update = x.times( y*(newAlpha-alpha(idx))/(lambda*n) )
        w = update.plus(w)
        deltaW = update.plus(deltaW)
        alpha(idx) = newAlpha
      }
    }

    val deltaAlpha = (alphaOld.times(-1.0)).plus(alpha)
    return (deltaAlpha, deltaW)
  }

}
