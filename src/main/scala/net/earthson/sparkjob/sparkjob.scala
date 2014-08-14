package net.earthson.sparkjob

import org.apache.spark._
import SparkContext._
import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator
import scala.util.Random
import scala.annotation.tailrec


abstract class SparkJob {
    private def uiport() = {
        val rd = new Random()
        @tailrec def random_port(x:Int):Int = {
            if(x > 1024) x 
            else random_port(rd.nextInt & 0xFFFF)
        }
        random_port(rd.nextInt & 0xFFFF)
    }
    @tailrec final def initspark(name:String):SparkContext = {
        val conf = new SparkConf()
                //.setMaster("yarn-client")
                .setMaster("yarn-cluster")
                .setAppName(name)
                .setSparkHome(System.getenv("SPARK_HOME"))
                .setJars(SparkContext.jarOfClass(this.getClass).toList)
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                //.set("spark.closure.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.kryoserializer.buffer.mb", "128")
                .set("spark.akka.frameSize", "256")
                //.set("spark.kryo.registrator", "net.earthson.nlp.MyRegistrator")
                .set("spark.default.parallelism", "32")
                //.set("spark.cleaner.ttl", "500")
                .set("spark.executor.memory", "4g")
                //.set("spark.ui.port", uiport().toString)
        val sc = new SparkContext(conf)
        if (sc == null) initspark(name) 
        else {
            sc.setCheckpointDir("hdfs://ns1/checkpoint_spark")
            sc
        }
    }

    def main(args: Array[String])
}
