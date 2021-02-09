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

}
