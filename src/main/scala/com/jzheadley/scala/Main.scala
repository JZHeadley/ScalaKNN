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

    val schema = DataTypes.createStructType(
      Array(
        StructField("label", IntegerType, nullable = false),
        StructField("features", ArrayType(DoubleType), nullable = false),
        StructField("id", LongType)
      )
    )

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val startTime = System.nanoTime()
    val spark = SparkSession.builder
      .appName("Scala KNN")
      .master("local[*]")
      .getOrCreate()
    log.info("Loading dataset")


    //    labelThroughSparkSQLTest(spark)

    val dataDfRaw = spark
      .read
      .format("org.apache.spark.ml.source.arff")
      .load(datasetPath)
      .withColumn("id", monotonically_increasing_id())
      .persist(StorageLevel.MEMORY_AND_DISK)

    val dataDf = spark
      .createDataFrame(
        dataDfRaw.rdd.map(row => {
          Row(row.getDouble(0).toInt, row.get(1).asInstanceOf[DenseVector].values.toSeq, row.getLong(2))
        }), schema)

    //    dataDf.show()
    dataDf.printSchema()
    //    val dataDf = readArffIntoDataFrame(spark, datasetPath, schema).persist(StorageLevel.MEMORY_AND_DISK)
    //    val data = dataDf.map(row => LabeledPoint(row.getInt(0), Vectors.dense(row.getSeq(1).asInstanceOf[Seq[Double]].toArray)))
    //    val Array(train, test) = data.randomSplit(Array(0.5, 0.5))
    //
    val k = 3
    val predictions = runKNNOnFullDataset(spark, dataDf, k)
    val accuracy = computeAccuracy(predictions)
    val diff = (System.nanoTime() - startTime) / 1000000.0
    //    val logisticRegressionAccuracy = computeAccuracy(runLogisticRegression(train, test))

    println(f"\nThe KNN classifier for ${dataDf.rdd.count} instances required $diff%01.4fms CPU time. Accuracy was $accuracy%01.4f\n")
    //    println(f"\nThe Logistic Regression classifier trained on ${train.rdd.count} instances and tested on ${test.rdd.count} instances had  Accuracy of  $logisticRegressionAccuracy%01.4f\n")

    spark.sqlContext.clearCache()
    spark.stop()

  }

  def labelThroughSparkSQLTest(spark: SparkSession): Unit = {
    import spark.implicits._
    val distances = Seq((1, 12.0), (1, 10.0), (0, 11.0), (1, 3.45), (2, 5.2), (6, 20.6), (7, 10.0), (7, 10.0), (6, 3.0))
      .toDF("label", "distance")
      .sort(desc("distance")) // sort it with highest distances first

    //    distances.show()
    println(vote(distances, 3))
  }

  import math._


  def runKNNOnFullDataset(spark: SparkSession, data: DataFrame, k: Int): DataFrame = {
    //    def runKNNOnFullDataset(spark: SparkSession, train: DataFrame, test: DataFrame, k: Int): DataFrame = {
    val schema = DataTypes.createStructType(
      Array(
        StructField("id", LongType),
        StructField("prediction", IntegerType, nullable = false)
      )
    )

    // Create an empty dataset


    //    emptyDF.union(predicted test instance)


    // Test dataframe using local iterator

    // For each test instance

    // Broadcast

    // UDF to find distances with the test instance and the train

    // sort distances

    // Pick k

    // Voting

    // Add to test instance

    // Predicted Union to emptyDF

    // Repeat


    // return emptyDF
    //    var test = data.where("id==1").first()
    //
    //    knn(spark, data.where("id!=1"), test.getSeq(1), 1, k)
    val emptyDf = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)

    import spark.implicits._
    val it = data.toLocalIterator()
    while (it.hasNext) {
      var row = it.next()
      var instanceId = row.getLong(2)
      println(instanceId)
      var prediction = knn(spark, data.where(s"id!=$instanceId"), row.getSeq(1), instanceId, k)
      var newRow = Seq((instanceId, prediction))
      //      newRow.show()
      emptyDf.union(newRow.toDF("id", "prediction"))
    }
    emptyDf
      .show()
    val ret = data.join(emptyDf, "id")
    ret
    //    data.withColumn("prediction", runKNNWithInstance(spark, k, data) (col("id")))
    //    data
  }

  def knn(spark: SparkSession, train: DataFrame, testFeatures: Seq[Double], testInstanceId: Long, k: Int): Int = {
    val distanceUDF = udf {
      features: Seq[Double] => distance(features, testFeatures)
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
