package net.earthson.nlp

import scala.collection.JavaConversions.asScalaBuffer
import org.apache.spark._
import SparkContext._
import scala.io.Source
import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator
import net.earthson.nlp.lda

class MyRegistrator extends KryoRegistrator {
    override def registerClasses(kryo: Kryo) {
        kryo.register(classOf[Map[String,Int]])
        kryo.register(classOf[Map[String,Long]])
        kryo.register(classOf[Seq[(String,Long)]])
        kryo.register(classOf[Seq[(String,Int)]])
    }
}

object ADLDAJob {
    val ntopics = 128
    def initspark(name:String) = {
        val conf = new SparkConf()
                    .setMaster("yarn-standalone")
                    .setAppName(name)
                    .setSparkHome(System.getenv("SPARK_HOME"))
                    .setJars(SparkContext.jarOfClass(this.getClass))
                    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                    //.set("spark.closure.serializer", "org.apache.spark.serializer.KryoSerializer")
                    .set("spark.kryoserializer.buffer.mb", "256")
                    .set("spark.kryo.registrator", "com.semi.nlp.MyRegistrator")
                    .set("spark.cores.max", "30")
                    .set("spark.default.parallelism", "30")
        new SparkContext(conf)
    }
    def main(args: Array[String]) {
        val spark = initspark("AD-LDA Testing")
        val file = spark.textFile("hdfs://ns1/nlp/lda/wiki.tuple3")
        val adldamodel = lda.ADLDAModel(spark, file, ntopics)
        adldamodel.train(100)
        spark.stop()
    }
}
