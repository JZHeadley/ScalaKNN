package com.jzheadley.scala

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object Main {

  def main(args: Array[String]) {
    val datasetPath = "/home/jzheadley/IdeaProjects/ScalaKNN/src/main/resources/small.arff"
    val log = Logger.getLogger(getClass.getName)
    log.info(datasetPath)
    val schema = DataTypes.createStructType(
      Array(
        StructField("features", ArrayType(DoubleType)),
        StructField("label", IntegerType)
      )
    )

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val startTime = System.nanoTime()
    val spark = SparkSession.builder
      .appName("Scala KNN")
      .master("local[*]")
      .getOrCreate()
    log.info("Loading ARFF dataset")


    val dataDf = readArffIntoDataFrame(spark, datasetPath, schema)
    //    dataDf.show()
    import spark.implicits._
    val data = dataDf.map(row => LabeledPoint(row.getInt(1), Vectors.dense(row(0).asInstanceOf[Seq[Double]].toArray)))
    //    data.foreach(point => {
    //      println(point.toString)
    //    })
    val Array(training, test) = data.randomSplit(Array(0.2, 0.8))
    //    val tokenizer = new Tokenizer()
    //      .setInputCol("Features")
    //      .setOutputCol("tokenized")
    //
    //    val hashingTf = new HashingTF()
    //      .setNumFeatures(dataDf.first().get(0).asInstanceOf[Seq[Double]].length)
    //      .setInputCol(tokenizer.getOutputCol)
    //      .setOutputCol("hashed")

    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.01)

    //    val pipeline = new Pipeline()
    //      .setStages(Array(tokenizer,hashingTf, lr))
    val model = lr.fit(training)

    val predictions = model.transform(test)
    // I could not get the metrics to work to save my life... then I remembered spark sql was a thing and I kinda enjoy this solution...
    val correctPredictions = predictions.where("prediction == label")
    correctPredictions.show(100)
    println(correctPredictions.rdd.count())
    println(predictions.rdd.count())

    val accuracy = correctPredictions.rdd.count.toFloat / predictions.rdd.count.toFloat
    val diff = (System.nanoTime() - startTime) / 1000000.0
    println(s"\nThe KNN classifier for ${dataDf.rdd.count} instances required ${diff}ms CPU time. Accuracy was $accuracy\n")

    spark.stop()

  }

  def readArffIntoDataFrame(spark: SparkSession, datasetPath: String, schema: StructType): DataFrame = {


    val dataRdd = spark.sparkContext.textFile(datasetPath)
      .map(line => line.toString.split(",")) // split everything by ,'s
      .filter(line => line.length > 1) // get rid of the arff header information rows
      .map(instance => {
      Row(instance.take(instance.length - 1).map(_.toDouble), instance.last.toInt) // map it to 2 columns so we have class as a separate column.  Hopefully it will help me maybe?
    })
    spark.createDataFrame(dataRdd, schema)
  }
}
