import AssemblyKeys._ // put this at the top of the file

name := "AD-LDA"

version := "0.1"

scalaVersion := "2.10.3"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.0.2" % "provided"

libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "2.2.0" % "provided"

libraryDependencies += "org.ansj" % "ansj_seg" % "1.4"

resolvers += "Akka Repository" at "http://repo.akka.io/releases/"

//resolvers += "Ansj Repository" at "http://maven.ansj.org/"

assemblySettings

assemblyOption in assembly ~= { _.copy(includeScala = false) }
