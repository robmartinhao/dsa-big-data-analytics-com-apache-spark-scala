package mlspark

/**
  * Clusterização - Segmentação de Veículos Para Transportadoras
  */

object SparkMLKMeansClustering extends App {

  // Import dos pacotes
  import org.apache.spark.sql.functions._
  import org.apache.log4j.Logger
  import org.apache.log4j.Level

  // Nível de log
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  // Variáveis
  val spSession = SparkUtils.SparkUtils.sparkSess
  val spContext = SparkUtils.SparkUtils.sparkCon
  val datadir = SparkUtils.SparkUtils.datadir

  // Carregando o arquivo em um RDD
  val autoData = spContext.textFile(datadir + "dataset4-autos.csv")
  autoData.cache()
  autoData.count()

  // Removendo a primeira linha (contendo cabeçalho)
  val firstLine = autoData.first()
  val dataLines = autoData.filter(x => x != firstLine)
  dataLines.count()

  import org.apache.spark.ml.linalg.Vectors
  import org.apache.spark.ml.feature.LabeledPoint
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.types._

  // Schema para o dataframe
  val schema =
    StructType(
      StructField("DOORS", DoubleType, false) ::
        StructField("BODY", DoubleType, false) ::
        StructField("HP", DoubleType, false) ::
        StructField("RPM", DoubleType, false) ::
        StructField("MPG", DoubleType, false) :: Nil)

  // Convertendo para um vetor
  def transformToNumeric( inputStr : String) : Row = {
    val attList = inputStr.split(",")

    val doors = attList(3).contains("two") match {
      case  true => 0.0
      case  false    => 1.0
    }
    val body = attList(4).contains("sedan") match {
      case  true => 0.0
      case  false    => 1.0
    }
    // Filtrando as colunas não necessárias nesta análise
    // Somente usaremos doors, body, hp, rpm, mpg-city
    val values= Row( doors, body,
      attList(7).toDouble, attList(8).toDouble,
      attList(9).toDouble)
    return values
  }

  // Convertendo para um vetor
  val autoVectors = dataLines.map(transformToNumeric)
  autoVectors.collect()

  println("Dados Transformados em Dataframe")
  val autoDf = spSession.createDataFrame(autoVectors, schema)
  autoDf.printSchema()
  autoDf.show(5)

}
