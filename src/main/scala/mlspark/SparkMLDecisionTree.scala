package mlspark

/**
  * Árvores de Decisão - Classificação de Plantas
  */

object SparkMLDecisionTree extends App {

  // Import dos pacotes
  import org.apache.log4j.Logger
  import org.apache.log4j.Level
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.types._
  import org.apache.spark.ml.linalg.Vectors
  import org.apache.spark.ml.feature.LabeledPoint
  import org.apache.spark.ml.classification.DecisionTreeClassifier
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
  import org.apache.spark.ml.feature.IndexToString


  // Log
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  // Variáveis
  val spSession = SparkUtils.SparkUtils.sparkSess
  val spContext = SparkUtils.SparkUtils.sparkCon
  val datadir = SparkUtils.SparkUtils.datadir

  // Carregando arquivo csv em um RDD
  println("\nCarregando o dataset...")
  val irisData = spContext.textFile(datadir + "dataset5-iris.csv")
  irisData.cache()
  irisData.take(5)

  // Removendo a primeira linha (contendo cabeçalho)
  val dataLines = irisData.filter(x =>  !x.contains("Sepal"))
  dataLines.count()

  // Convertendo o RDD em um Vetor Denso

  // Schema para o Dataframe
  val schema =
    StructType(
      StructField("SPECIES", StringType, false) ::
        StructField("SEPAL_LENGTH", DoubleType, false) ::
        StructField("SEPAL_WIDTH", DoubleType, false) ::
        StructField("PETAL_LENGTH", DoubleType, false) ::
        StructField("PETAL_WIDTH", DoubleType, false) :: Nil)

  def transformToNumeric( inputStr : String) : Row = {
    val attList = inputStr.split(",")

    // Filtrando as Colunas
    val values= Row(attList(4),
      attList(0).toDouble,
      attList(1).toDouble,
      attList(2).toDouble,
      attList(3).toDouble
    )
    return values
  }

  // Ajustando o vetor
  val irisVectors = dataLines.map(transformToNumeric)
  irisVectors.collect()

  println("\nDados Transformados em Dataframe")
  val irisDf = spSession.createDataFrame(irisVectors, schema)
  irisDf.printSchema()
  irisDf.show(5)

  // Indexação é um pré-requisito para Decision Trees
  import org.apache.spark.ml.feature.StringIndexer

  val stringIndexer = new StringIndexer()
  stringIndexer.setInputCol("SPECIES")
  stringIndexer.setOutputCol("INDEXED")

  val si_model = stringIndexer.fit(irisDf)
  val indexedIris = si_model.transform(irisDf)
  indexedIris.show()
  indexedIris.groupBy("SPECIES","INDEXED").count().show()

  println("\nAnálise de Correlação :")
  for ( field <- schema.fields ) {
    if ( ! field.dataType.equals(StringType)) {
      println("Correlação entre INDEXADO e " + field.name + " = " + indexedIris.stat.corr("INDEXED", field.name))
    }
  }

  // Transformando o dataframe e removendo colunas não utilizadas
  def transformToLabelVectors(inStr : Row ) : LabeledPoint = {
    val labelVectors = new LabeledPoint(
      inStr.getDouble(5) ,
      Vectors.dense(inStr.getDouble(1),
        inStr.getDouble(2),
        inStr.getDouble(3),
        inStr.getDouble(4)));
    return labelVectors
  }
  val tempRdd1 = indexedIris.rdd.repartition(2);
  val irisLabelVectors = tempRdd1.map(transformToLabelVectors)
  irisLabelVectors.collect()

  val irisDf2 = spSession.createDataFrame(
    irisLabelVectors, classOf[LabeledPoint] )

  println("\nDados Prontos Para ML")
  irisDf2.select("label","features").show(10)


  // Divisão em treino e teste
  val Array(trainingData, testData)
  = irisDf2.randomSplit(Array(0.7, 0.3))
  trainingData.count()
  testData.count()

  // Criando o Modelo
  val dtClassifier = new DecisionTreeClassifier()
  dtClassifier.setMaxDepth(2)
  val dtModel = dtClassifier.fit(trainingData)

  dtModel.numNodes
  dtModel.depth

  val rawPredictions = dtModel.transform(testData)

  // Convertendo os índices indexados de volta para o original
  val labelConverter = new IndexToString()
    .setInputCol("label")
    .setOutputCol("labelStr")
    .setLabels(si_model.labels)

  val predConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictionStr")
    .setLabels(si_model.labels)

  val predictions = predConverter.transform(labelConverter.transform(rawPredictions))
  println("\nPrevisões :")
  predictions.show()

  val evaluator = new MulticlassClassificationEvaluator()
  evaluator.setPredictionCol("prediction")
  evaluator.setLabelCol("label")
  evaluator.setMetricName("accuracy")
  println("\nAcurácia = " + evaluator.evaluate(predictions)  )

  println("\nConfusion Matrix:")

  // Desenhando a Confusion Matrix
  predictions.groupBy("labelStr","predictionStr").count().show()

}
