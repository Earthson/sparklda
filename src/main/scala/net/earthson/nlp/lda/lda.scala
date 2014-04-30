package net.earthson.nlp.lda

import scala.annotation.tailrec

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.Logging

import scala.collection.mutable
import net.earthson.nlp.sampler.MultiSampler


class ModelInfo(val ntopics:Int, val nterms:Int) {
    val alpha = ntopics/50.0
    val beta = 0.01
}

class TopicInfo(val nzw:Seq[((Int,String),Long)], val nz:Seq[(Int, Long)]) extends Serializable{}


object LDA {
    type LDADataRDD = RDD[(Long,String,Int)]

    def topicInfo(mwz:LDA.LDADataRDD) = {
        val nzw = new rdd.PairRDDFunctions(mwz.map(x=>((x._3, x._2),1L))).reduceByKey(_+_).map(x=>(x._1,x._2)).collect.toSeq
        val nz = nzw.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum.toLong).toSeq
        new TopicInfo(nzw, nz) 
    }

    def fromFile(sc:SparkContext, datapath:String, npart:Int = 100):LDADataRDD = {
        sc.textFile(datapath, npart).map(x=>{
                val tmp = x.split("\t")
                (tmp(0).toLong, tmp(1), tmp(2).toInt)
            })
    }

    def toFile(mwz:LDADataRDD, datapath:String) {
        mwz.saveAsTextFile(datapath)
    }

    def topWords(tinfo:TopicInfo, limit:Int = 20):Seq[(Int, Seq[(String, Long)])] = {
        tinfo.nzw.groupBy(_._1._1).map(x=>(x._1,x._2.map(y=>(y._1._2, y._2)).toSeq.sortBy(_._2).reverse)).toSeq.sortBy(_._1)
    }

    def gibbsMapper(modelInfo: ModelInfo, topicinfo:TopicInfo, mwz:Seq[(Long,String,Int)], round:Int) = {
        import modelInfo._
        println("mapping")
        val mwzbuf = mwz.toBuffer
        val nzw = mutable.Map(topicinfo.nzw:_*).withDefaultValue(0)
        val nz = mutable.Map(topicinfo.nz:_*).withDefaultValue(0)
        val nmz = mutable.Map(mwz.groupBy(x=>(x._1, x._3)).mapValues(_.size).toSeq:_*).withDefaultValue(0)
        val nm = mutable.Map(nmz.toSeq.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum).toSeq:_*).withDefaultValue(0)
        val probz = new Array[Double](ntopics)
        for(r <- 1 to round) {
            printf("Iteration: %d\n", r)
            for(i <- 0 until mwzbuf.size) {
                val (cm, cw, cz) = mwzbuf(i)
                nzw((cz, cw)) -= 1
                nz(cz) -= 1
                nmz((cm,cz)) -= 1
                //nm(cm) -= 1
                for(z <- 0 until ntopics) probz(z) = ((nzw((z,cw))+beta)/(nz(z)+nterms*beta))*(nmz((cm,z))+alpha)
                val newz = MultiSampler.multisampling(probz)
                mwzbuf(i) = (cm, cw, newz)
                nzw((newz,cw)) += 1
                nz(newz) += 1
                nmz((cm,newz)) += 1
                //nm(cm) += 1
            }
        }
        mwzbuf.toSeq
    }
}

class ADLDA(
        val ntopics:Int,
        val nterms:Int, 
        val npartition:Int=100) 
    extends Serializable with Logging
{
    //TODO: flexible save location and input location
    //TODO: Add Perplexity

    //private val nterms = fromFile(datapath).map(_._2).distinct.count.toInt
    private val modelinfo = new ModelInfo(ntopics, nterms)

    def train(imwz:LDA.LDADataRDD, round:Int, innerRound:Int = 20):LDA.LDADataRDD = {
        val mwzToRun = if (imwz.partitions.size >= this.npartition) imwz else imwz.repartition(this.npartition)
        val minfo = mwzToRun.sparkContext broadcast this.modelinfo
        @tailrec def loop(i:Int, mwz:LDA.LDADataRDD):LDA.LDADataRDD = {
            if(i == round) mwz
            else {
                val tinfo = mwz.sparkContext broadcast LDA.topicInfo(mwz)
                val mwzNew = mwz.mapPartitions(
                        it=>LDA.gibbsMapper(minfo.value, tinfo.value, it.toSeq, innerRound).toIterator, 
                        preservesPartitioning=true)
                    .persist
                loop(i+1, mwzNew)
            }
        }
        loop(1, mwzToRun)
    }
}

class LDALocal(var mwz:Seq[(Long,String,Int)], val ntopics:Int) {
    //TODO: Add Perplexity

    val nterms = Set(mwz.map(_._2):_*).size
    val modelinfo = new ModelInfo(ntopics, nterms)

    def nzw = mwz.groupBy(x=>(x._3, x._2)).mapValues(_.size.toLong).toSeq
    def nz = nzw.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum.toLong).toSeq
    def tinfo = new TopicInfo(this.nzw, this.nz)
    
    def train(round:Int) {
        this.mwz = LDA.gibbsMapper(this.modelinfo, this.tinfo, this.mwz, round)
    }

    def twords(limit:Int) = {
        val info = this.nzw.groupBy(_._1._1).map(x=>(x._1,x._2.map(y=>(y._1._2, y._2)).toSeq.sortBy(_._2).reverse)).toSeq.sortBy(_._1)
        for((tp, tpw) <- info) {
            printf("%d\t:%s\n", tp, tpw.take(limit).map(_._1).mkString(sep="\t"))
        }
    }
}
