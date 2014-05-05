package net.earthson.nlp.lda

import scala.annotation.tailrec

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.Logging
import org.apache.spark.storage.StorageLevel._

import scala.collection.mutable
import net.earthson.nlp.sampler.MultiSampler


class ModelInfo(val ntopics:Int, val nterms:Int) extends Serializable{
    val alpha = ntopics/50.0
    val beta = 0.01
}

class TopicInfo(val nzw:Seq[((Int,String),Long)], val nz:Seq[(Int, Long)]) extends Serializable{}


object LDA {
    //TODO: Add Perplexity

    type LDADataRDD = RDD[List[(String,Int)]]
    type LDADoc = Seq[List[(String,Int)]]

    def topicInfo(mwz:LDA.LDADataRDD) = {
        val nzw = new rdd.PairRDDFunctions(mwz.flatMap(xl=>xl.map(x=>((x._2,x._1),1L)))).reduceByKey(_+_).collectAsMap.toSeq
        val nz = nzw.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum.toLong).toSeq
        new TopicInfo(nzw, nz) 
    }

    def fromFile(sc:SparkContext, datapath:String, ntopics:Int, npart:Int = 100):LDADataRDD = {
        def readPair(wz:String) = {
            val tmp = wz.split("""/-""", 2)
            (tmp(0), tmp(1).toInt)
        }
        sc.textFile(datapath, npart).map(xl=> {
            val rd = new scala.util.Random;
            xl.split("""\s+""").map(w=>if(w contains "/-") readPair(w) else (w, rd.nextInt(ntopics))).toList
        })
    }

    def toFile(mwz:LDADataRDD, datapath:String) {
        mwz.map(xl=>xl.map(x=>s"${x._1}/-${x._2}").mkString(sep="\t")).saveAsTextFile(datapath)
    }

    def loadADLDA(sc:SparkContext, datapath:String, ntopics:Int, npart:Int = 100):ADLDAModel = {
        val mwz = LDA.fromFile(sc, datapath, ntopics, npart)
        val ntc = mwz.flatMap(_.map(_._1)).distinct.count.toInt
        new ADLDAModel(ntopics, ntc, mwz, npart)
    }

    def topWords(tinfo:TopicInfo, limit:Int = 20):Seq[(Int, Seq[(String, Long)])] = {
        tinfo.nzw.groupBy(_._1._1).map(x=>(x._1,x._2.map(y=>(y._1._2, y._2)).toSeq.sortBy(_._2).reverse)).toSeq.sortBy(_._1)
    }

    def gibbsMapper(modelInfo: ModelInfo, topicinfo:TopicInfo, omwz:Seq[List[(String,Int)]], round:Int, inference:Boolean=false) = {
        import modelInfo._
        val mwz = omwz.map(_.toArray)
        val nzw = mutable.Map(topicinfo.nzw:_*).withDefaultValue(0)
        val nz = new Array[Long](ntopics)
        for((z, c) <- topicinfo.nz) nz(z) = c
        val nmz = mwz.map(mdoc=>{
                        val marr = new Array[Long](ntopics)
                        for( (w, z) <- mdoc) {
                            marr(z) += 1
                        }
                        marr
                    })
        val probz = new Array[Double](ntopics)
        for(r <- 1 to round) {
            for((mdoc, docmz) <- mwz zip nmz) {
                for(((cw, cz), i) <- mdoc.zipWithIndex) {
                    if (inference == false) {
                        nzw((cz, cw)) -= 1
                        nz(cz) -= 1
                    }
                    docmz(cz) -= 1
                    for(z <- 0 until ntopics) probz(z) = ((nzw((z,cw))+beta)/(nz(z)+nterms*beta))*(docmz(z)+alpha)
                    val newz = MultiSampler.multisampling(probz)
                    mdoc(i) = (cw, newz)
                    if (inference == false) {
                        nzw((newz,cw)) += 1
                        nz(newz) += 1
                    }
                    docmz(newz) += 1
                }
            }
        }
        mwz.map(_.toList)
    }
}


class ADLDAModel (
        final val ntopics:Int,
        final val nterms:Int, 
        var data:LDA.LDADataRDD,
        final val npartition:Int=100) 
{

    val minfo = new ModelInfo(ntopics, nterms)
    var tinfo = LDA.topicInfo(this.data)

    def train(round:Int, innerRound:Int = 20) {
        this.data = if (this.data.partitions.size >= this.npartition) this.data else this.data.repartition(this.npartition)
        val minf = this.minfo
        @tailrec def loop(i:Int, mwz:LDA.LDADataRDD, tpinfo:TopicInfo):(LDA.LDADataRDD, TopicInfo) = {
            println(s"Round:\t${i}")
            if(i == round) (mwz, tpinfo)
            else {
                val tinfo = mwz.sparkContext broadcast tpinfo
                val mwzNew = mwz.mapPartitions(
                        it=>LDA.gibbsMapper(minf, tinfo.value, it.toSeq, innerRound).toIterator, 
                                    preservesPartitioning=true)
                    .persist(MEMORY_AND_DISK)
                val tpinfoNew = LDA.topicInfo(mwzNew)
                mwz.unpersist(blocking=true)
                loop(i+1, mwzNew, tpinfoNew)
            }
        }
        val (x, y) = loop(1, this.data, this.tinfo)
        this.data = x
        this.tinfo = y
    }

    def inference(infer:LDA.LDADataRDD, round:Int = 20, npart:Int = 0) = {
        val todo = if(npart > 0) infer.repartition(npart) else infer
        val minf = this.minfo
        val tinfo = infer.sparkContext broadcast this.tinfo
        infer.mapPartitions(it=>LDA.gibbsMapper(minf, tinfo.value, it.toSeq, round, inference=true).toIterator, 
                                    preservesPartitioning=true)
    }
}

