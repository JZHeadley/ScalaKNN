package com.jzheadley.scala

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ListBuffer

object Main {

  def main(args: Array[String]) {
    val smallDatasetPath = "/home/jzheadley/IdeaProjects/ScalaKNN/src/main/resources/small.arff"
    val mediumDatasetPath = "/home/jzheadley/IdeaProjects/ScalaKNN/src/main/resources/medium.arff"
    val datasetPath = smallDatasetPath
    val log = Logger.getLogger(getClass.getName)
    log.info(datasetPath)

    val schema = DataTypes.createStructType(
      Array(
        StructField("label", IntegerType, nullable = false),
        StructField("features", ArrayType(DoubleType), nullable = false),
        StructField("id", LongType)
      )
    )

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("Scala KNN")
      .master("local[*]")
      .getOrCreate()
    log.info("Loading dataset")


    //    labelThroughSparkSQLTest(spark)

    val dataDf = spark
      .read
      .format("org.apache.spark.ml.source.arff")
      .load(datasetPath)
      .withColumn("id", monotonically_increasing_id())
      .persist(StorageLevel.MEMORY_AND_DISK)


    //    dataDf.printSchema()
    val k = 3
    val startTime = System.nanoTime()
    val predictions = runKNNOnFullDataset(spark, dataDf, k)
    predictions.show()
    val accuracy = computeAccuracy(predictions)
    val diff = (System.nanoTime() - startTime) / 1000000.0

    println(f"\nThe KNN classifier for ${dataDf.rdd.count} instances required $diff%01.4fms CPU time. Accuracy was $accuracy%01.4f\n")

    spark.sqlContext.clearCache()
    spark.stop()
  }

  def runKNNOnFullDataset(spark: SparkSession, data: DataFrame, k: Int): DataFrame = {
    val predictions = ListBuffer[(Long, Double)]()
    import spark.implicits._
    val it = data.toLocalIterator()
    while (it.hasNext) {
      val row = it.next()
      val instanceId = row.getLong(2)
      println(instanceId)
      val prediction = knn(spark, data.where(s"id!=$instanceId"), row.get(1).asInstanceOf[DenseVector], instanceId, k)
      predictions.append((instanceId, prediction))
    }
    data.join(predictions.seq.toDF("id", "prediction"), "id")
  }

  def knn(spark: SparkSession, train: DataFrame, testFeatures: DenseVector, testInstanceId: Long, k: Int): Double = {
    val distanceUDF = udf {
      features: DenseVector => Vectors.sqdist(testFeatures, features)
    }
    val distanced = train.withColumn("distance", distanceUDF(train("features")))
      // just take the labels column since its all we need now doing it here might make things faster since we're dealing with less data?
      .select(col("label"), col("distance"))
      .sort(desc("distance")) // sort it with highest distances first
    //    distanced.show()
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
  def vote(distanced: DataFrame, k: Int): Double = {
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
        .getDouble(0) // get the label from the row so we can return it as the prediction
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
    correctPredictions.rdd.count.toFloat / predictions.rdd.count.toFloat
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
