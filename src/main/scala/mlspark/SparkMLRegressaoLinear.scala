package mlspark

/**
 * Regressão Linear - Prevendo Autonomia de Combustível em Veículos
 * http://archive.ics.uci.edu/ml/datasets/auto+mpg
 */

object SparkMLRegressaoLinear extends App {

  import org.apache.log4j.Logger
  import org.apache.log4j.Level
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.types._
  import org.apache.spark.ml.linalg.Vectors
  import org.apache.spark.ml.feature.LabeledPoint
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.ml.regression.LinearRegression

  // Configurando o nível de log
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  // Variáveis definidas a partir do arquivo de utilitários
  val spSession = SparkUtils.SparkUtils.sparkSess
  val spContext = SparkUtils.SparkUtils.sparkCon
  val datadir = SparkUtils.SparkUtils.datadir

  //  Carregando o arquivo csv em um RDD
  val autoData = spContext.textFile(datadir + "dataset3-mpg.csv")
  autoData.cache()

  // Removendo a primeira linha (contendo cabeçalho)
  val dataLines = autoData.filter(x => !x.contains("CYLINDERS"))
  println("\nTotal de linhas no dataset : " + dataLines.count())

  // Converte o RDD em um Vetor Denso
  //   1. Removemos colunas que não são necessárias
  //   2. Alteramos valores não-numéricos ( values=? ) para numéricos
  //Use default for average HP

  // Schema para o dataframe
  val schema =
    StructType(
      StructField("MPG", DoubleType, false) ::
        StructField("CYLINDERS", DoubleType, false) ::
        StructField("HP", DoubleType, false) ::
        StructField("ACCELERATION", DoubleType, false) ::
        StructField("MODELYEAR", DoubleType, false) :: Nil)

  val avgHP = spContext.broadcast(80.0)

  // Função para transformar dados em valores numéricos
  def transformToNumeric( inputStr : String) : Row = {

    val attList = inputStr.split(",")

    // Replace dos valores ? por valores normais
    var hpValue = attList(3)
    if (hpValue.contains("?")) {
      hpValue = avgHP.value.toString
    }
    // Filtrando colunas que não são necessárias nesta análise
    val values = Row((attList(0).toDouble ),
      attList(1).toDouble,
      hpValue.toDouble,
      attList(5).toDouble,
      attList(6).toDouble
    )
    return values
  }
  // Mantemos somente MPG, CYLINDERS, HP,ACCELERATION and MODELYEAR
  val autoVectors = dataLines.map(transformToNumeric)

  println("\nDados Limpos e Transformados : " )
  autoVectors.collect()

  // Converte para Dataframe
  val autoDf = spSession.createDataFrame(autoVectors, schema)
  autoDf.show(5)

  // Análise de Correlação
  println("\nAnálise de Correlação : " )
  for ( field <- schema.fields ) {
    if ( ! field.dataType.equals(StringType)) {
      println("Correlação entre MPG e "
        + field.name +
        " = " + autoDf.stat.corr(
        "MPG", field.name))
    }
  }

  // Transformação para um dataframe a ser usado como entrada para o modelo
  // Remoção de colunas que não são necessárias (baixa correlação)
  def transformToLabelVectors(inStr : Row ) : LabeledPoint = {
    val labelVectors = new LabeledPoint(
      inStr.getDouble(0) ,
      Vectors.dense(inStr.getDouble(1),
        inStr.getDouble(2),
        inStr.getDouble(3),
        inStr.getDouble(4)));
    return labelVectors
  }

  val tempRdd1 = autoDf.rdd.repartition(2);
  val autoLabelVectors = tempRdd1.map(transformToLabelVectors)
  autoLabelVectors.collect()

  val autoDF = spSession.createDataFrame(autoLabelVectors, classOf[LabeledPoint] )

  println("\nDados Pré-Processados : " )
  autoDF.select("label","features").show(10)

  // Divisão em Treino e Teste
  val Array(trainingData, testData) = autoDF.randomSplit(Array(0.9, 0.1))
  println("\nDados de Treino - Count : " + trainingData.count())
  println("\nDados de Teste - Count : " + testData.count())
  println("\n")

  // Construindo o modelo nos dados de treino
  val lr = new LinearRegression().setMaxIter(10)
  val lrModel = lr.fit(trainingData)

  println("\nCoeficientes: " + lrModel.coefficients)
  print("\nIntercepto: " + lrModel.intercept)
  lrModel.summary.r2

  // Previsões nos dados de teste
  val predictions = lrModel.transform(testData)
  println("\nPrevisões : " )
  predictions.select("prediction","label","features").show()

  // Avaliando o Resultado e encontrado o coeficiente R2
  val evaluator = new RegressionEvaluator()
  evaluator.setPredictionCol("prediction")
  evaluator.setLabelCol("label")
  evaluator.setMetricName("r2")
  println("\nAcurácia = " + evaluator.evaluate(predictions))

}
