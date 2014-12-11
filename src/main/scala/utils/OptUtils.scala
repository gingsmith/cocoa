package distopt.utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.collection.immutable.SortedMap
import scala.util._
import java.io._
import scala.io.Source

object OptUtils {

  // load data stored in LIBSVM format and append line ID
  def loadLIBSVMData(sc: SparkContext, filename: String, numSplits: Int, numFeats: Int): RDD[SparseClassificationPoint] = {

    // read in text file
    val data = sc.textFile(filename,numSplits).coalesce(numSplits)  // note: coalesce can result in data being sent over the network. avoid this for large datasets
    val numEx = data.count()

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit{ case(i,lines) =>
      Iterator(i -> lines.length)
    }.collect().sortBy(_._1)
    val offsets = sizes.map(x => x._2).scanLeft(0)(_+_).toArray

    // parse input
    data.mapPartitionsWithSplit { case(partition, lines) =>
      lines.zipWithIndex.flatMap{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        if(index < numEx){

          // parse label
          val parts = line.trim().split(' ')
          var label = -1
          if (parts(0).contains("+") || parts(0).toInt == 1)
            label = 1

          // parse features
          var features = new SparseVector(SortedMap(parts.slice(1,parts.length)
               .map(_.split(':') 
               match { case Array(i,j) => (i.toInt-1,j.toDouble)}):_*))

          // create classification point
          Iterator(SparseClassificationPoint(index,label,features))
        }
        else{
          Iterator()
        }
      }
    }
  }

  def loadImageNetData(sc: SparkContext, filename: String, nsplits: Int): RDD[SparseClassificationPoint] = {

    // read in text file
    val data = sc.textFile(filename,nsplits)
    val c = data.first.split(",")(0)

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit{ case(i,lines) =>
      Iterator(i -> lines.length)
    }.collect().toMap
    val offsets = sizes.values.scanLeft(0)(_+_).toArray

    // parse input
    data.mapPartitionsWithSplit { case(partition, lines) =>
      lines.zipWithIndex.map{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        // parse label
        val parts = line.trim().split(",")
        val label = if(parts(0)==c) 1 else -1

        val features = new SparseVector(SortedMap((1 to parts.length-1)
          .map( a => (a.toInt-1,parts(a).toDouble)).filter{ case(a,b) => b!=0.0 }:_*))

        // create classification point
        SparseClassificationPoint(index,label,features)
      }
    }

  }

  // dot product of two dense arrays
  def dotDense(dense1: Array[Double], dense2: Array[Double]) : Double = {
    return dense1.zipWithIndex.map{ case(v,i) => v*dense2(i) }.sum
  }

  // find norm of dense arrays
  def normDense(dense: Array[Double]): Double = {
    dotDense(dense,dense)
  }

  // calculate hinge loss for a point (label,features) given a weight vector
  def hingeLoss(point: SparseClassificationPoint, w:Array[Double]) : Double = {
    val y = point.label
    val X = point.features
    return Math.max(1 - y*(X.dot(w)),0.0)
  }

  // compute hinge gradient for a point (label, features) given a weight vector
  def computeHingeGradient(point: SparseClassificationPoint, w: Array[Double]) : SparseVector = {
    val y = point.label
    val X = point.features
    val eval = 1 - y * (X.dot(w))
    if(eval > 0){
      return X.times(y)
    }
    else{
      return new SparseVector(SortedMap.empty[Int,Double])
    }
  }

  // compute dual loss
  def dualLoss(d:(SparseClassificationPoint,SparseClassificationPoint), a:Array[Double]) : Double = {
    val qij = d._1.features.dot(d._2.features)
    return .5*a(d._1.index)*a(d._2.index)*d._1.label*d._2.label*qij
  }

  // can be used to compute train or test error
  def computeAvgLoss(data: RDD[SparseClassificationPoint], w: Array[Double]) : Double = {
    val n = data.count()
    return data.map(hingeLoss(_,w)).reduce(_+_) / n
  }

  /**
   * Compute the primal objective function value.
   * Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
   * 
   * @param data
   * @param w
   * @param n
   * @param lambda
   * @return
   */
  def computePrimalObjective(data: RDD[SparseClassificationPoint], w: Array[Double], lambda: Double): Double = {
    return (computeAvgLoss(data, w) + (normDense(w) * lambda * 0.5))
  }

  /**
   * Compute the dual objective function value.
   * Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
   *
   * @param data
   * @param w
   * @param n
   * @param lambda
   * @return
   */
  def computeDualObjective(data: RDD[SparseClassificationPoint], w: Array[Double], alpha : RDD[(Int, Double)], lambda: Double): Double = {
    val n = data.count()
    val sumAlpha = alpha.map(kv => kv._2).reduce(_ + _)
    return (-lambda / 2 * OptUtils.normDense(w) + sumAlpha / n)
  }
  /**
   * Compute the duality gap value.
   * Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
   *
   * @param data
   * @param w
   * @param alpha
   * @param lambda
   * @return
   */
  def computeDualityGap(data: RDD[SparseClassificationPoint], w: Array[Double], alpha: RDD[(Int, Double)], lambda: Double): Double = {
    return (computePrimalObjective(data, w, lambda) - computeDualObjective(data, w, alpha, lambda))
  }

  def computeClassificationError(data: RDD[SparseClassificationPoint], w:Array[Double]) : Double = {
    val n = data.count()
    return data.map(x => if((x.features).dot(w)*(x.label) > 0) 0.0 else 1.0).reduce(_ + _)/n
  }

  def printSummaryStatsPrimalDual(algName: String, data: RDD[SparseClassificationPoint], w: Array[Double], alpha: RDD[(Int, Double)], lambda: Double, testData: RDD[SparseClassificationPoint]) = {
    var outString = algName + " has finished running. Summary Stats: "
    val objVal = computePrimalObjective(data, w, lambda)
    outString = outString + "\n Total Objective Value: " + objVal
    val dualityGap = computeDualityGap(data, w, alpha, lambda)
    outString = outString + "\n Duality Gap: " + dualityGap
    if(testData!=null){
      val testErr = computeClassificationError(testData, w)
      outString = outString + "\n Test Error: " + testErr
    }
    println(outString + "\n")
  }

  def printSummaryStats(algName: String, data: RDD[SparseClassificationPoint], w: Array[Double], lambda: Double, testData: RDD[SparseClassificationPoint]) =  {
    var outString = algName + " has finished running. Summary Stats: "
    val objVal = computePrimalObjective(data, w, lambda)
    outString = outString + "\n Total Objective Value: " + objVal
    if(testData!=null){
      val testErr = computeClassificationError(testData, w)
      outString = outString + "\n Test Error: " + testErr
    }
    println(outString + "\n")
  }
  
}
