package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")

    val df : DataFrame = spark.read
      .option("header",true)
      .option("inferSchema","true")
      .csv("data/train_clean.csv")

    print(s"Nombre de ligne :  ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")
    df.show()
    df.printSchema()
    val dfCasted : DataFrame = df
      .withColumn("goal", df("goal").cast("Int"))
      .withColumn("deadline" , df("deadline").cast("Int"))
      .withColumn("state_changed_at", df("state_changed_at").cast("Int"))
      .withColumn("created_at", df("created_at").cast("Int"))
      .withColumn("launched_at", df("launched_at").cast("Int"))
      .withColumn("backers_count", df("backers_count").cast("Int"))
      .withColumn("final_status", df("final_status").cast("Int"))

    dfCasted.printSchema()

    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy(dfCasted("count").desc).show(100)
    dfCasted.groupBy("currency").count.orderBy(dfCasted("count").desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy(dfCasted("count").desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy(dfCasted("count").desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy(dfCasted("count").desc).show(50)

    val df2: DataFrame = dfCasted.drop("disable_communication")
  }
}
