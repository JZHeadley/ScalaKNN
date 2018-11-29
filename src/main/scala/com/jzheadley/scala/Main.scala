package com.jzheadley.scala

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.DenseVector
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


    labelThroughSparkSQLTest(spark)


    val dataDf = spark
      .read
      .format("org.apache.spark.ml.source.arff")
      .load(datasetPath)
      .withColumn("id", monotonically_increasing_id)
      .persist(StorageLevel.MEMORY_AND_DISK)
    dataDf.show()
    dataDf.collect()
    //    val dataDf = readArffIntoDataFrame(spark, datasetPath, schema).persist(StorageLevel.MEMORY_AND_DISK)
    import spark.implicits._
    val data = dataDf.map(row => LabeledPoint(row.getDouble(0), row.get(1).asInstanceOf[DenseVector]))
    val Array(train, test) = data.randomSplit(Array(0.5, 0.5))


    val predictions = runKNNOnFullDataset(spark, dataDf, 3)
    val accuracy = computeAccuracy(predictions)
    val diff = (System.nanoTime() - startTime) / 1000000.0
    val logisticRegressionAccuracy = computeAccuracy(runLogisticRegression(train, test))

    println(f"\nThe KNN classifier for ${dataDf.rdd.count} instances required $diff%01.4fms CPU time. Accuracy was $accuracy%01.4f\n")
    println(f"\nThe Logistic Regression classifier trained on ${train.rdd.count} instances and tested on ${test.rdd.count} instances had  Accuracy of  $logisticRegressionAccuracy%01.4f\n")

    spark.sqlContext.clearCache()
    spark.stop()

  }

  def labelThroughSparkSQLTest(spark: SparkSession) = {
    import spark.implicits._
    val distances = Seq(1, 1, 0, 1, 2, 6, 7, 7, 6).toDF("label")
    distances.show()
    val countLabels = distances
      .groupBy("label")
      .agg(count("label").as("labelCounts"))
      .limit(3)

    countLabels.show()
    val prediction = countLabels
      .first()
      .getInt(0)
    println(prediction)
  }

  import math._

  def runKNNOnFullDataset(spark: SparkSession, data: DataFrame, k: Int): DataFrame = {
    val knnUDF = udf { id: Long =>
      val test = data.where(s"id==$id")
      //      val test: DataFrame = instances.filter(row => row.getLong(2) == id)
      println(test)
      //      test.show()
      knn(spark, data.where(s"id!=$id"), test.first().get(1).asInstanceOf[Seq[Double]], id, k)
    }


    data.withColumn("prediction", knnUDF(data("id")))

    //    data.foreach(test => {
    //      val testId = test.getLong(2)
    //      //      val train = data
    //      //        .filter(row => {
    //      //          println(row)
    //      //          row.getLong(2) != testId
    //      //        })
    //      //        .collect()
    //      //      println(testId)
    //      //      println(test.get(1))
    //      val testFeatures = test.get(1).asInstanceOf[DenseVector].values.toSeq
    //      //      println(testFeatures)
    //      knn(spark, data, testFeatures, testId, k)
    //  }

    //  )
  }

  def knn(spark: SparkSession, train: DataFrame, testFeatures: Seq[Double], testInstanceId: Long, k: Int): Int = {
    val distanceUDF = udf {
      features: DenseVector =>
        println(features)
        println(testFeatures)
        //          0
        distance(features.values.toSeq, testFeatures)
    }

    //    val distances = train.map(row => {
    //      println(row)
    //
    //      (distance(testFeatures, row(1).asInstanceOf[Seq[Double]]), row(1).asInstanceOf[Int])
    //    })
    //    val blah = distanceUDF(train("features"))
    val distanced = train.withColumn("distance", distanceUDF(train("features")))


    distanced.show()
    // because spark does magic this hopefully isn't incredibly slow... probably is anyway
    // is that groupby with an aggregate n^2 ? feel like it might be... good thing k isn't generally very large
    distanced.sort(desc("distance")) // sort it with highest distances first
      .limit(k) // take the k closest
      .select(col("label")) // just take the labels column since its all we need now
      .groupBy("label") // groupby the label
      .agg(count("label").as("labelCounts")) //count how many instances of each label we have
      .first() // take the first row which should have the highest occurrence
      .getInt(0) // get the label from the row so we can return it as the prediction
    //    0
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
      .setMaxIter(10)
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
