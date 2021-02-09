package mlspark

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

object SparkUtils {

  /**
   * Definição de utilitários usados pelos demais algoritmos de Machine Learning com Scala e Spark deste capítulo
   */

  object SparkUtils {

    // Diretório onde estão os datasets
    val datadir = "/home/robson/Projects/dsa-big-data-analytics-com-apache-spark-scala/datasets"

    // Nome da Instância Spark (pode ser qualquer nome)
    val appName = "SparkML"

    // Local ou Url da Instância Spark
    val sparkMasterURL = "local[2]"

    // Diretório temporário requerido pelo Spark SQL
    // No Windows, altere o caminho do diretório
    val tempDir = "/tmp/spark-warehouse"

    var sparkSess: SparkSession = null
    var sparkCon: SparkContext = null

    // Inicialização  - Executa quando o objeto é criado
    {
      // É necessário configurar hadoop.home.dir para evitar erros na inicialização do Spark
      // No Windows, altere o caminho do diretório
      System.setProperty("hadoop.home.dir",
        "/tmp/hadoop")

      // Objeto de configuração do Spark
      val conf = new SparkConf()
        .setAppName(appName)
        .setMaster(sparkMasterURL)
        .set("spark.executor.memory", "2g")
        .set("spark.sql.shuffle.partitions", "2")

      // Get ou create Spark context. Cria uma nova instância se não existir uma disponível.
      sparkCon = SparkContext.getOrCreate(conf)

      // Cria sessão Spark SQL
      sparkSess = SparkSession
        .builder()
        .appName(appName)
        .master(sparkMasterURL)
        .config("spark.sql.warehouse.dir", tempDir)
        .getOrCreate()
    }
  }
}
