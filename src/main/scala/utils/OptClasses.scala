package distopt.utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

// Dense Classification Point
case class ClassificationPoint(val label: Double, val features: Array[Double])

// Sparse Classifcation Point
case class SparseClassificationPoint(val index: Int, val label: Double, val features: SparseVector)

// Dense Regression Point
case class RegressionPoint(val label: Double, val features: Array[Double])

// Sparse Regression Point
case class SparseRegressionPoint(val label: Double, val features: SparseVector)

// Sparse Vector Implementation
class SparseVector(val indices: Array[Int], val values: Array[Double]) extends Serializable{

  def apply(index: Int): Double = values.apply(indices.indexOf(index))

  def getOrElse(index: Int, default: Double): Double = {
    if(this.indices.contains(index)){
      return this.values(indices.indexOf(index))
    } 
    return default
  }

  // scale a sparse vector by a constant
  def times(c: Double) : SparseVector = {
    return new SparseVector(this.indices,this.values.map(x => x*c))
  }

  // add this sparse vector to another sparse vector, return sparse vector
  def plus(sparse:SparseVector) : SparseVector = {
    val combined = sparse.indices.zip(sparse.values) ++ this.indices.zip(this.values)
    val sumArr = combined.groupBy( _._1).map { case (k,v) => k -> v.map(_._2).sum }.toArray
    return new SparseVector(sumArr.map(x => x._1), sumArr.map(x => x._2))
  }

  // add this sparse vector to a dense vector, return dense vector
  def plus(dense:Array[Double]) : Array[Double] = {
    this.indices.zipWithIndex.foreach{ case(idx,i) => (dense(idx) = dense(idx)+this.values(i))}
    return dense
  }

  // dot product of sparse vector (sortedmap) and dense vector
  def dot(dense: Array[Double]) : Double = {
    var total = 0.0
    this.indices.zipWithIndex.foreach{ case(idx,i) => (total += this.values(i)*dense(idx)) }
    return total
  }

  // dot product of two sparse vectors
  def dot(sparse: SparseVector) : Double = {
    var total = 0.0
    this.indices.zipWithIndex.foreach{ case(idx,i) => (total += this.values(i)*sparse.getOrElse(idx,0.0))}
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