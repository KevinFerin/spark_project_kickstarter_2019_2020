package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{datediff, hour, to_date, udf,lower,concat,lit, from_unixtime,format_number,when}
import org.apache.spark.sql.types.{DateType}

object Preprocessor {

  def cleanCountry(country: String, currency: String): String = {
    if (country == "False" )
      currency
    else
      country
  }

  def cleanCurrency(currency: String): String = {
    if (currency != null && currency.length != 3)
      null
    else
      currency
  }

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

    import spark.implicits._


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
    //df.show()
    //df.printSchema()
    val dfCasted : DataFrame = df
      .withColumn("goal", df("goal").cast("Int"))
      .withColumn("deadline" , df("deadline").cast("Int"))
      .withColumn("state_changed_at", df("state_changed_at").cast("Int"))
      .withColumn("created_at", df("created_at").cast("Int"))
      .withColumn("launched_at", df("launched_at").cast("Int"))
      .withColumn("backers_count", df("backers_count").cast("Int"))
      .withColumn("final_status", df("final_status").cast("Int"))

    //dfCasted.printSchema()

    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show
    /*
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(20)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(20)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(20)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(20)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(20)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)
    */
    val df2: DataFrame = dfCasted.drop("disable_communication")

    val dfNoFutur : DataFrame = df2.drop("backers_count","state_changed_at")

    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    //dfCountry.groupBy("final_status").count.orderBy($"count".desc).show()
    // Ici nous allons séléctionner que les campagens ayant un final-status à 0 ou 1.
    // On pourrait toutefois tester en mettant toutes les autres valeurs à 0
    // en considérant que les campagnes qui ne sont pas un Success sont un Fail.
    //val cleanFinalStatusUdf = udf(cleanFinalStatus _)
    val dfFinalStatus : DataFrame = dfCountry
      .withColumn("final_status", when($"final_status"===0 || $"final_status"===1,$"final_status").otherwise(null))
        .filter($"final_status".isNotNull)
    //dfFinalStatus.groupBy("final_status").count.orderBy($"count".desc).show()
    //dfFinalStatus.printSchema()
   // dfFinalStatus.show()

    val dfNbDays : DataFrame = dfFinalStatus
      .withColumn("deadline2",from_unixtime($"deadline"))
      .withColumn("launched_at2",from_unixtime($"launched_at"))
      .withColumn("created_at2",from_unixtime($"created_at"))
      .withColumn("days_campaign", datediff($"deadline2",$"launched_at2"))
      .withColumn("hours_prepa", format_number(($"launched_at" - $"created_at")/3600,3))
      .drop("launched_at","created_at","deadline","launched_at2","created_at2","deadline2")

    //dfNbDays.show()
    val dfText : DataFrame = dfNbDays
      .withColumn("desc", lower($"desc"))
      .withColumn("name", lower($"name"))
      .withColumn("keywords", lower($"keywords"))
        .withColumn("text",concat($"name",lit(" "),$"desc",lit(" "),$"keywords"))
        .drop("name","desc","keywords")
    //dfText.show()
    //val cleanNullIntUdf = udf(cleanNullInt _)
    //val cleanNullStringUdf = udf(cleanNullString _)

    val dfCleanNull : DataFrame = dfText
      .withColumn("days_campaign",when($"days_campaign".isNull,-1).otherwise($"days_campaign"))
      .withColumn("goal",when($"goal".isNull, -1).otherwise($"goal"))
      .withColumn("hours_prepa",when($"hours_prepa".isNull,-1).otherwise(($"hours_prepa")))
      .withColumn("country2",when($"country2"==="True","unknown").otherwise($"country2"))
      .withColumn("currency2",when($"currency2".isNull , "unknown").otherwise($"currency2"))

    //dfCleanNull.show()
    dfCleanNull.groupBy("country2").count().orderBy($"count").show()
    dfCleanNull.groupBy("currency2").count().orderBy($"count").show()
    //dfCleanNull.groupBy("days_campaign").count().orderBy($"count").show()
    //Environ 22000 ligne avec des hours_prepa negatifs et que 37 en dessous de -10 370 en dessous de -5 => solution de les mettre tous à 0
    //print("ceci est un test pre clean : ",dfCleanNull.filter($"hours_prepa" === -1).count())
    val dfCleaned : DataFrame = dfCleanNull
        .withColumn("hours_prepa", when($"hours_prepa" < 0, -1).otherwise($"hours_prepa"))
    //print("ceci est un test post clean ",dfCleaned.filter($"hours_prepa" < -5).count())
    print(dfCleanNull.filter($"days_campaign" <= 0).count())
    //peux pas sauvegarder, faut enlever les utf et utiliser les fonctions spark : DONE
    //////dfCleanNull.write.parquet("~/Documents/Telecom/Intro au framework SPARK/spark_project_kickstarter_2019_2020/TP2parquet")
  }

}
