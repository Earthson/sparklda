package net.earthson.nlp

import scala.collection.JavaConversions.asScalaBuffer
import org.apache.spark._
import SparkContext._
import scala.io.Source
import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator

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
                    .set("spark.kryoserializer.buffer.mb", "64")
                    .set("spark.akka.frameSize", "128")
                    .set("spark.kryo.registrator", "net.earthson.nlp.MyRegistrator")
                    .set("spark.default.parallelism", "32")
                    //.set("spark.cores.max", "30")
                    //.set("spark.executor.memory", "1g")
        new SparkContext(conf)
    }

    def main(args: Array[String]) {
        val spark = initspark("AD-LDA Testing")
        val adldaModel = lda.LDA.loadADLDA(spark, "hdfs://ns1/nlp/lda/wiki.docs.10000", 64)
        adldaModel.train(round=100, innerRound=10)
        for((tp, tpw) <- lda.LDA.topWords(adldaModel.tinfo)) {
            printf("%d\t:\t%s\n", tp, tpw.take(20).map(_._1).mkString(sep="\t"))
        }
        spark.stop()
    }
}
