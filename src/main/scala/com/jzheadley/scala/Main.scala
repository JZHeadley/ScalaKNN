package com.jzheadley.scala

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object Main {
  def main(args: Array[String]) {
    val datasetPath = "/home/jzheadley/IdeaProjects/ScalaKNN/src/main/resources/small.arff"
    val log = Logger.getLogger(getClass.getName)
    log.info(datasetPath)

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val startTime = System.nanoTime()
    val spark = SparkSession.builder
      .appName("Scala KNN")
      .master("local[*]")
      .getOrCreate()
    log.info("Loading ARFF dataset")


    val dataDf = readArffIntoDataFrame(spark, datasetPath)
    dataDf.show()

    val diff = (System.nanoTime() - startTime) / 1000000.0
    val accuracy = 0.0
    println(s"\nThe KNN classifier for ${dataDf.rdd.count} instances required ${diff}ms CPU time. Accuracy was $accuracy\n")

    spark.stop()

  }

  def readArffIntoDataFrame(spark: SparkSession, datasetPath: String): DataFrame = {
    val schema = DataTypes.createStructType(
      Array(
        StructField("Features", ArrayType(DoubleType)),
        StructField("Class", IntegerType)
      )
    )

    val dataRdd = spark.sparkContext.textFile(datasetPath)
      .map(line => line.toString.split(",")) // split everything by ,'s
      .filter(line => line.length > 1) // get rid of the arff header information rows
      .map(instance => {
      Row(instance.map(_.toDouble), instance.last.toInt) // map it to 2 columns so we have class as a separate column.  Hopefully it will help me maybe?
    })
    return spark.createDataFrame(dataRdd, schema)
  }
}
