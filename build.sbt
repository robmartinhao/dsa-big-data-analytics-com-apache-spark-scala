name := "dsa-big-data-analytics-com-apache-spark-scala"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)
