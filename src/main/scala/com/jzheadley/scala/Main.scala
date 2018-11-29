package com.jzheadley.scala

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

object Main {

  def main(args: Array[String]) {
    val smallDatasetPath = "/home/jzheadley/IdeaProjects/ScalaKNN/src/main/resources/small.arff"
    val mediumDatasetPath = "/home/jzheadley/IdeaProjects/ScalaKNN/src/main/resources/medium.arff"
    val datasetPath = smallDatasetPath
    val log = Logger.getLogger(getClass.getName)
    log.info(datasetPath)
    //    val schema = DataTypes.createStructType(
    //      Array(
    //        StructField("features", ArrayType(DoubleType)),
    //        StructField("label", IntegerType)
    //      )
    //    )

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val startTime = System.nanoTime()
    val spark = SparkSession.builder
      .appName("Scala KNN")
      .master("local[*]")
      .getOrCreate()
    log.info("Loading dataset")

    val dataDf = spark
      .read
      .format("org.apache.spark.ml.source.arff")
      .option("header", "true")
      .load(datasetPath)
      .withColumn("id", monotonically_increasing_id)
      .persist(StorageLevel.MEMORY_AND_DISK)
    dataDf.show()
    dataDf.collect()
    //    val dataDf = readArffIntoDataFrame(spark, datasetPath, schema) //.persist(StorageLevel.MEMORY_AND_DISK)
    import spark.implicits._
    val data = dataDf.map(row => LabeledPoint(row.getInt(0), Vectors.dense(row(1).asInstanceOf[Seq[Double]].toArray)))
    val Array(train, test) = data.randomSplit(Array(0.2, 0.8))


    //    val predictions = runLogisticRegression(train, test)
    val predictions = runKNNOnFullDataset(spark, dataDf, 3)
    //    val accuracy = computeAccuracy(predictions)
    val diff = (System.nanoTime() - startTime) / 1000000.0
    //    val logisticRegressionAccuracy = computeAccuracy(runLogisticRegression(train, test))

    //    println(f"\nThe KNN classifier for ${dataDf.rdd.count} instances required $diff%01.4fms CPU time. Accuracy was $accuracy%01.4f\n")
    //    println(f"\nThe Logistic Regression classifier trained on ${train.rdd.count} instances and tested on ${test.rdd.count} instances had  Accuracy of  $logisticRegressionAccuracy%01.4f\n")

    spark.sqlContext.clearCache()
    spark.stop()

  }

  import math._

  def runKNNOnFullDataset(spark: SparkSession, data: DataFrame, k: Int): Unit = {
    //    val instances = data.select("*")
    //    val test = data.first()
    //    val knnUDF = udf { id: Long =>
    //      println(id)
    //      //      val test = instances.where(s"id==$id")
    ////      val test: DataFrame = instances.filter(row => row.getLong(2) == id)
    //      println(test)
    //      //      test.show()
    //      //      knn(spark, data.where(s"id!=$id"), test.first().get(1).asInstanceOf[Seq[Double]], id, k)
    //      0
    //    }
    //

    //    data.withColumn("prediction", predictions)
  }

  def knn(spark: SparkSession, train: DataFrame, testFeatures: Seq[Double], testInstanceId: Long, k: Int): Int = {
    val distanceUDF = udf { features: Seq[Double] =>
      distance(features, testFeatures)
    }
    //    val distances = train.map(row => {
    //      println(row)
    //
    //      (distance(testFeatures, row(1).asInstanceOf[Seq[Double]]), row(1).asInstanceOf[Int])
    //    })
    train.withColumn("distance", distanceUDF(train("features")))
    train.show()
    //      .sort(col("distance"))
    0
  }

  def distance(features1: Seq[Double], features2: Seq[Double]): Double = {
    sqrt(
      (features1 zip features2).map {
        case (x, y) => pow(y - x, 2)
      }.sum
    )
  }

  /*
   * I couldn't get the metrics to work to save my life...
   * Then I remembered spark sql was a thing and came up with this method...
   * I kinda liked the fact that I could do it so I didn't bother to figure
   * out what was wrong with the metrics.
   */
  def computeAccuracy(predictions: DataFrame): Double = {
    val correctPredictions = predictions.where("prediction == label")
    correctPredictions.rdd.count.toFloat / predictions.rdd.count.toFloat
  }

  def runLogisticRegression(train: Dataset[LabeledPoint], test: Dataset[LabeledPoint]): DataFrame = {
    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.01)
    val model = lr.fit(train)
    model.transform(test)
  }

  /*
   * I wrote this method when I couldn't get the arff library to work
   */
  def readArffIntoDataFrame(spark: SparkSession, datasetPath: String, schema: StructType): DataFrame = {
    val dataRdd = spark.sparkContext.textFile(datasetPath)
      .map(line => line.toString.split(",")) // split everything by ,'s
      .filter(line => line.length > 1) // get rid of the arff header information rows
      .map(instance => {
      Row(instance.take(instance.length - 1).map(_.toDouble), instance.last.toInt) // map it to 2 columns so we have class as a separate column.  Hopefully it will help me maybe?
    })

    spark.createDataFrame(dataRdd, schema)
      .withColumn("id", monotonically_increasing_id)
  }
}
