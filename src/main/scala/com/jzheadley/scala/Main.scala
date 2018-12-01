package com.jzheadley.scala

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ListBuffer

object Main {

  def displayMetrics(predictionAndLabels: RDD[(Double, Double)]): Unit = {
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

  }

  def main(args: Array[String]) {

    val smallDatasetPath = "/home/jzheadley/IdeaProjects/ScalaKNN/src/main/resources/small.arff"
    //    val mediumDatasetPath = "/home/jzheadley/IdeaProjects/ScalaKNN/src/main/resources/medium.arff"
    var datasetPath = smallDatasetPath
    //            datasetPath = mediumDatasetPath
    val log = Logger.getLogger(getClass.getName)
    log.info(datasetPath)

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("Scala KNN")
      .master("local[*]")
      .getOrCreate()
    log.info("Loading dataset")
    //    labelThroughSparkSQLTest(spark)


    // If I use Jorge's library I can't seem to get the label correctly and only read it in as a 0
    // As a result I always predict 0 which gives me 100% accuracy so thats fun...
    //    val dataDf = spark
    //      .read
    //      .format("org.apache.spark.ml.source.arff")
    //      .load(datasetPath)
    //      .withColumn("id", monotonically_increasing_id())
    val dataDf: DataFrame = readArffIntoDataFrame(spark, datasetPath)
      .persist(StorageLevel.MEMORY_AND_DISK)
    //    dataDf.show(400)
    //    dataDf.printSchema()
    val k = 3
    val startTime = System.nanoTime()
    val predictions = runKNNOnFullDataset(spark, dataDf, k)
    val diff = (System.nanoTime() - startTime) / 1000000.0
    val accuracy = computeAccuracy(predictions)
    //    displayMetrics(predictions.rdd.map(row => (row.getInt(1).toDouble, row.getInt(3).toDouble)))

    // should get 79.17% on small dataset and 60.62% on medium
    println(f"\nThe KNN classifier for ${dataDf.rdd.count} instances required $diff%01.4fms CPU time. Accuracy was $accuracy%01.2f%%\n")

    spark.sqlContext.clearCache()
    spark.stop()
  }

  def runKNNOnFullDataset(spark: SparkSession, data: DataFrame, k: Int): DataFrame = {
    val predictions = ListBuffer[(Long, Int)]()
    import spark.implicits._
    val it = data.toLocalIterator()
    while (it.hasNext) {
      val row = it.next()
      val instanceId = row.getLong(2)
      //      println(instanceId)
      val prediction = knn(spark, data.where(s"id!=$instanceId"), row.get(1).asInstanceOf[DenseVector], instanceId, k)
      predictions.append((instanceId, prediction))
    }
    data.join(predictions.seq.toDF("id", "prediction"), "id")
  }

  def knn(spark: SparkSession, train: DataFrame, testFeatures: DenseVector, testInstanceId: Long, k: Int): Int = {
    val distanceUDF = udf {
      features: DenseVector => Vectors.sqdist(testFeatures, features)
    }
    val distanced = train.withColumn("distance", distanceUDF(train("features")))
      // just take the labels column since its all we need now doing it here might make things faster since we're dealing with less data?
      .select(col("label"), col("distance"))
      .sort("distance") // sort it with lowest distances first
    vote(distanced, k)
  }

  /**
    * Knn Voting algorithm with tie breaking created almost exclusively with Spark SQL...
    * Really hope Spark SQL is in some way optimized and is part of the magic...
    *
    * @param distanced dataframe with a distance and a label column presorted in descending order by distance
    * @param k         the current k closest we're looking for
    * @return the most common class amongst the k closest
    */
  def vote(distanced: DataFrame, k: Int): Int = {
    // because spark does magic this hopefully isn't incredibly slow... probably is anyway
    // is that groupby with an aggregate n^2 ? feel like it might be... good thing k isn't generally very large
    val labelCounts = distanced
      .limit(k) // take the k closest
      .groupBy("label") // groupby the label
      .agg(count("label").as("labelCounts")) //count how many instances of each label we have
    // if we don't have k distinct values there is a tie in voting and we should reduce k and revote
    if (labelCounts.select("labelCounts").distinct().count() != k) {
      vote(distanced, k - 1) //yay functional programming!
    }
    else {
      labelCounts
        .first() // take the first row which should have the highest occurrence
        .getInt(0) // get the label from the row so we can return it as the prediction
    }
  }

  /*
   * I couldn't get the metrics to work to save my life...
   * Then I remembered spark sql was a thing and came up with this method...
   * I kinda liked the fact that I could do it so I didn't bother to figure
   * out what was wrong with the metrics.
   */
  def computeAccuracy(predictions: DataFrame): Double = {
    val correctPredictions = predictions.where("prediction == label")
    (correctPredictions.rdd.count.toFloat / predictions.rdd.count.toFloat) * 100.0
  }


  /*
   * I wrote this method when I couldn't get the arff library to work
   */
  def readArffIntoDataFrame(spark: SparkSession, datasetPath: String): DataFrame = {
    val schema = DataTypes.createStructType(
      Array(
        StructField("label", IntegerType),
        StructField("featuresSeq", ArrayType(DoubleType))
      )
    )
    val dataRdd = spark.sparkContext.textFile(datasetPath)
      .map(line => line.toString.split(",")) // split everything by ,'s
      .filter(line => line.length > 1) // get rid of the arff header information rows
      .map(instance =>
      Row(instance.last.toInt, instance.take(instance.length - 1).map(_.toDouble)) // map it to 2 columns so we have class as a separate column.  Hopefully it will help me maybe?
    )

    val vectorizeUDF = udf {
      features: Seq[Double] => Vectors.dense(features.toArray)
    }
    spark.createDataFrame(dataRdd, schema)
      .withColumn("features", vectorizeUDF(col("featuresSeq")))
      .withColumn("id", monotonically_increasing_id)
      .select("label", "features", "id")
  }
}
