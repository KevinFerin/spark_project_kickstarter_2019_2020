package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.ml.feature.{IDF, Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer, OneHotEncoder, VectorAssembler,CountVectorizerModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit,TrainValidationSplitModel, CrossValidator}
object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    val df : DataFrame = spark.read
      .parquet("data/prepared_trainingset")

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokensWOstopwords")

    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("tokensWOstopwords")
      .setOutputCol("countedWord")
      .setMinDF(2) //a word has to appear 2 times to be in the vocabulary
      .fit(stopWordsRemover.transform(tokenizer.transform(df)))

    val idf = new IDF()
      .setInputCol("countedWord")
      .setOutputCol("tfidf")

    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val onehotencoderCountry = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")

    val onehotencoderCurrency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    val splits = df.randomSplit(Array(0.9, 0.1))
    val training = splits(0).cache()
    val test = splits(1)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover,cvModel,idf, indexerCountry,indexerCurrency,
        onehotencoderCountry, onehotencoderCurrency, assembler, lr))

    val model = pipeline.fit(training)

    model.write.overwrite().save("spark-logistic-regression-model")

    val sameModel = PipelineModel.load("spark-logistic-regression-model")

    val predic = sameModel.transform(test)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")
    val result = evaluator.evaluate(predic)

    println("\n")
    println("Le f1 score de ce model sans tuning est : " + result)
    println("\n")


    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, Array(55.0,75.0,95.0))
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    val lrtv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)

    val modelGridTV = lrtv.fit(training)

    modelGridTV.write.overwrite().save("spark-logistic-regression-model-gridSearchedTV")

    val sameModelGridTV = TrainValidationSplitModel.load("spark-logistic-regression-model-gridSearchedTV")

    val predicGridTV = sameModelGridTV.transform(test)

    val resultGridTV = evaluator.evaluate(predicGridTV)

    println("\n")
    println("Le f1 score de ce model après gridSearch avec train validation split est : " + resultGridTV)
    println("\n")

    val lrcv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(5)

    val modelGridCV = lrcv.fit(training)

    modelGridCV.write.overwrite().save("spark-logistic-regression-model-gridSearchedCV")

    val sameModelGridCV = TrainValidationSplitModel.load("spark-logistic-regression-model-gridSearchedCV")

    val predicGridCV = sameModelGridCV.transform(test)

    val resultGridCV = evaluator.evaluate(predicGridCV)

    println("\n")
    println("Le f1 score de ce model après gridSearch avec cross validator est : " + resultGridCV)
    println("\n")
  }
}
