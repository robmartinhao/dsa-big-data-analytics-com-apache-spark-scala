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


  // Clustering - Center e Scaling

  val meanVal = autoDf.agg(avg("DOORS"), avg("BODY"), avg("HP"), avg("RPM"), avg("MPG")).collectAsList().get(0)

  val stdVal = autoDf.agg(stddev("DOORS"), stddev("BODY"), stddev("HP"),stddev("RPM"),stddev("MPG")).collectAsList().get(0)

  val bcMeans = spContext.broadcast(meanVal)
  val bcStdDev = spContext.broadcast(stdVal)

  def centerAndScale(inRow : Row ) : LabeledPoint  = {
    val meanArray = bcMeans.value
    val stdArray = bcStdDev.value

    var retArray = Array[Double]()

    for (i <- 0 to inRow.size - 1)  {
      val csVal = ( inRow.getDouble(i) - meanArray.getDouble(i)) /
        stdArray.getDouble(i)
      retArray = retArray :+ csVal
    }
    return  new LabeledPoint(1.0,Vectors.dense(retArray))
  }

  val tempRdd1 = autoDf.rdd.repartition(2);
  val autoCSRDD = tempRdd1.map(centerAndScale)
  autoCSRDD.collect()

  val autoDf2 = spSession.createDataFrame(autoCSRDD, classOf[LabeledPoint] )

  println("Dados Prontos Para o Modelo")
  autoDf2.select("label","features").show(10)

  import  org.apache.spark.ml.clustering.KMeans
  val kmeans = new KMeans()
  kmeans.setK(4)
  kmeans.setSeed(1L)

  // Criando Modelo K-Means para Clusterização
  val model = kmeans.fit(autoDf2)
  val predictions = model.transform(autoDf2)

  println("Grupos :")
  predictions.select("features","prediction").show()
  predictions.groupBy("prediction").count().show()

}
