package mlspark

/**
 * Manipulação de Dados com Spark SQL
 */

object SparkSQL extends App {

  // Importando pacotes
  import org.apache.log4j.Logger
  import org.apache.log4j.Level
  import org.apache.spark.sql.functions._

  // Definindo nível de log
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  // Definindo variáveis do Spark a partir do SparkUtils
  val spSession = SparkUtils.SparkUtils.sparkSess
  val spContext = SparkUtils.SparkUtils.sparkCon
  val datadir = SparkUtils.SparkUtils.datadir

  // Trabalhando com Dataframes

  // Criando um Dataframe a partir de um arquivo JSON
  val empDf = spSession.read.json(datadir + "dataset1-clientes.json")
  empDf.show()
  empDf.printSchema()

  // Select
  empDf.select("nome", "salario").show()

  // Filtro
  empDf.filter(empDf("idade") === 60).show()

  // Group By
  empDf.groupBy("sexo").count().show()

  // Aggregate
  empDf.groupBy("deptid").agg(avg(empDf("salario")), max(empDf("idade"))).show()

  // Criando um Dataframe a partir de uma lista
  val deptList = Array("{'nome': 'Vendas', 'id': '1000'}", "{ 'nome':'Controladoria','id':'2000' }")

  // Convertendo a lista para RDD
  val deptRDD = spContext.parallelize(deptList)

  // Carregando Dataframe no RDD
  val deptDf = spSession.read.json(deptRDD)
  deptDf.show()

  // Join entre dataframes
  empDf.join(deptDf, empDf("deptid") === deptDf("id")).show()

  // Cascateamento de Operações
  empDf.filter(empDf("idade") > 30).join(deptDf, empDf("deptid") === deptDf("id")).groupBy("deptid").agg(avg("salario"), max("idade")).show()


  // Criando um dataframe a partir de arquivo csv
  val autoDf = spSession.read.option("header", "true").csv(datadir + "dataset2-autos.csv")

  autoDf.show(5)

  // Criando dataframe a partir de um schema

  import org.apache.spark.sql.Row
  import org.apache.spark.sql.types._

  val schema =
    StructType(
      StructField("id", IntegerType, false) ::
        StructField("nome", StringType, false) :: Nil)

  val productList = Array((1001, "Book"), (1002, "Perfume"))

  val prodRDD = spContext.parallelize(productList)

  prodRDD.count()
  prodRDD.collect()


  def transformToRow(input: (Int, String)): Row = {

    // Filtrando as colunas
    val values = Row(input._1, input._2)
    return values
  }

  val prodRDD2 = prodRDD.map(transformToRow)

  val prodDf = spSession.createDataFrame(prodRDD2, schema)
  prodDf.show()
}

