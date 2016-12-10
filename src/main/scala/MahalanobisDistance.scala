import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.SQLContext
import com.databricks.spark.csv.CsvContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.collect_set
import org.apache.spark.sql.Row
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.stat.{ MultivariateStatisticalSummary, Statistics }
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import java.util.concurrent.{ Future, Callable, Executors }
import org.apache.commons.math3.stat.StatUtils
import scala.collection.mutable.ListBuffer
import breeze.linalg._
import breeze.numerics._
import breeze.numerics

object MahalabonisDistance {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Mahalanobis Distance");
    val sc = new SparkContext(conf);
    val sqlContext = new SQLContext(sc);
    val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/home/dhaval/spark/data/alertSample.csv");
    // df.show(); 	
    val assembler = new VectorAssembler().setInputCols(Array("WA_Date_In_Service", "WA_Mileage", "VIN_UniqueNum", "WA_AbsNum", "WA_Cost", "PQR_AbsNum", "PA_Warranty_EWI", "PA_DMS_EWI", "EH_SCORE", "EH_DMS_SCORE", "WA_Accumulated_Cost", "WA_AccumulatedAbsNum", "VIN_Accumulated_UniqueNum", "CE_ImpSafety", "CE_GenSafety", "CE_ImpGood")).setOutputCol("features")
    val output = assembler.transform(df)
    // println(output.select("features", "alert_id").first())
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(output)
    val scaledData = scalerModel.transform(output)
    //scaledData.show()
    val dataArray = scaledData.rdd.map(_.getAs[org.apache.spark.ml.linalg.DenseVector]("scaledFeatures")).map(a => org.apache.spark.mllib.linalg.Vectors.dense(a.values).toArray)
    val evMatrix = DenseMatrix(dataArray.collect: _*)
    val t1 = new RowMatrix(scaledData.rdd.map(_.getAs[org.apache.spark.ml.linalg.DenseVector]("scaledFeatures")).map(a => org.apache.spark.mllib.linalg.Vectors.dense(a.values)))
    val t3 = t1.computeColumnSummaryStatistics()
    val mean = DenseVector(t3.mean.toArray)
    val t2 = t1.computeCovariance()
    val newCov = new DenseMatrix[Double](t2.numRows, t2.numCols, t2.toArray)
    val dev: DenseMatrix[Double] = evMatrix(*, ::) - mean
    val dev: DenseMatrix[Double] = evMatrix(*, ::) - mean
    val dev: DenseMatrix[Double] = evMatrix(*, ::) - mean
    val mahaResult = sqrt(sum((dev * newCov) :* dev, Axis._1))
    val mahaDF = sc.parallelize(mahaResult.toArray).toDF("MahaDistance")
    val newDF = df.withColumn("uniqueID", monotonicallyIncreasingId)
    val mahaNewDF = mahaDF.withColumn("uniqueID", monotonicallyIncreasingId)
    val finalCSV = newDF.join(mahaNewDF, newDF("uniqueID") === mahaNewDF("uniqueID"), "fullouter").drop("uniqueID")
  }
}