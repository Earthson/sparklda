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
        kryo.register(classOf[lda.TopicInfo])
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
                    .set("spark.kryo.registrator", "net.earthson.nlp.MyRegistrator")
                    .set("spark.cores.max", "30")
                    .set("spark.default.parallelism", "30")
                    .set("spark.executor.memory", "4g")
        new SparkContext(conf)
    }
    def main(args: Array[String]) {
        val spark = initspark("AD-LDA Testing")
        val file = spark.textFile("hdfs://ns1/nlp/lda/wiki.tuple3.10000", 20)
        val ldardd = file.map(x=>{
                val tmp = x.substring(1, x.length-1).split(",")
                (tmp(0).toLong, tmp(1), tmp(2).toInt)
            })
        val adldamodel = new lda.ADLDAModel(ntopics, ldardd)
        adldamodel.train(spark, ldardd, 100)
        spark.stop()
    }
}
