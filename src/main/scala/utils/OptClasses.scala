package distopt.utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.collection.immutable.SortedMap

// Dense Classification Point
case class ClassificationPoint(val label: Double, val features: Array[Double])

// Sparse Classifcation Point
case class SparseClassificationPoint(val index: Int, val label: Double, val features: SparseVector)

// Dense Regression Point
case class RegressionPoint(val label: Double, val features: Array[Double])

// Sparse Regression Point
case class SparseRegressionPoint(val label: Double, val features: SparseVector)

// Sparse Vector Implementation
class SparseVector(val data: SortedMap[Int,Double]) extends Serializable{

  def apply(index: Int): Double = data.apply(index)

  def iterator = data.iterator

  // scale a sparse vector by a constant
  def times(c: Double) : SparseVector = {
    return new SparseVector(SortedMap(this.data.mapValues(_*c).toArray:_ *))
  }

  // add this sparse vector to another sparse vector, return sparse vector
  def plus(sparse:SparseVector) : SparseVector = {
    val list = sparse.data.toList ++ this.data.toList
    return new SparseVector(SortedMap(list.groupBy ( _._1) .map { case (k,v) => k -> v.map(_._2).sum }.toArray:_*))
  }

  // add this sparse vector to a dense vector, return dense vector
  def plus(dense:Array[Double]) : Array[Double] = {
    this.data.foreach{ case(i,v) => (dense(i) = dense(i)+v)}
    return dense
  }

  // dot product of sparse vector (sortedmap) and dense vector
  def dot(dense: Array[Double]) : Double = {
    var total = 0.0
    this.data.foreach{ case(i,v) => (total += v*dense(i)) }
    return total
  }

  // dot product of two sparse vectors
  def dot(sparse: SparseVector) : Double = {
    var total = 0.0
    this.data.foreach{ case(i,v) => (total += v*sparse.data.getOrElse(i,0.0).asInstanceOf[Double]) }
    return total
  }
}

class DoubleArray(arr: Array[Double]){
  def plus(plusArr: Array[Double]) : Array[Double] = {
    val retArr = (0 to plusArr.length-1).map( i => this.arr(i) + plusArr(i)).toArray
    return retArr
  }
  def times(c: Double) : Array[Double] = {
    val retArr = this.arr.map(x => x*c)
    return retArr
  }
}

object Implicits{
  implicit def arraytoDoubleArray(arr: Array[Double]) = new DoubleArray(arr)
}