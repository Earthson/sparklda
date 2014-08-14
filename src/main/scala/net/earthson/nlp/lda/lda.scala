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

class TopicInfo(val nzw:Map[(Int,String),Long], val nz:Map[Int, Long]) extends Serializable{}

class DocInfo(val nmz:Map[Int,Long], val nm:Long) extends Serializable{}


object LDA {

    type LDADataRDD = RDD[List[(String,Int)]]
    type LDADoc = Seq[List[(String,Int)]]

    def genTopicInfo(mwz:LDA.LDADataRDD) = {
        val nzw = new rdd.PairRDDFunctions(mwz.flatMap(xl=>xl.map(x=>((x._2,x._1),1L)))).reduceByKey(_+_).collect.toMap.withDefaultValue(0L)
        val nz = nzw.foldLeft(Map[Int,Long]().withDefaultValue(0L)) {(m,x)=>m.updated(x._1._1,m(x._1._1)+x._2)}
        //assert(nzw.map(_._2).sum == nz.map(_._2).sum)
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

    def genNMZ(mwz:Seq[List[(String,Int)]], ntopics:Int) = {
        mwz.map(mdoc=>{
                        val marr = new Array[Long](ntopics)
                        for( (w, z) <- mdoc) {
                            marr(z) += 1
                        }
                        marr
                    })
    }

    def genDocInfo(doc:List[(String, Int)]) = {
        val mp = doc.map(_._2).foldLeft(Map[Int,Long]().withDefaultValue(0L))((z,a)=>z+(a->(z(a)+1)))
        new DocInfo(mp, doc.size)
    }

    def genGlobalDocInfo(mwz:LDADataRDD) = {
        val mp = new rdd.PairRDDFunctions(mwz.flatMap(_.map(_._2)).map((_,1L))).reduceByKey(_+_).collectAsMap.toMap.withDefaultValue(0L)
        val msize = mwz.map(_.size).reduce(_+_)
        new DocInfo(mp, msize)
    }

    def entropySum(modelInfo: ModelInfo, topicInfo:TopicInfo, mwz:Seq[List[(String,Int)]]) = {
        import modelInfo._
        val nzw = Map(topicInfo.nzw.toSeq:_*).withDefaultValue(0L)
        val nz = new Array[Long](ntopics)
        for((z,c) <- topicInfo.nz) nz(z) = c
        def docH(mdoc:List[(String,Int)]) = {
            val mz = Array.fill(ntopics)(0L)
            mdoc.foreach(x=>mz(x._2) += 1)
            val nm = mz.sum
            def pw(w:String):Double = (0 until ntopics).map(z=> ((nzw((z, w))+beta)/(nz(z)+nterms*beta)*(mz(z)+alpha)/(nm+ntopics*alpha))).sum
            def eUnit(p:Double) = -Math.log(p)/Math.log(2)
            //val docEnt = mdoc.map(w => {assert(pw(w._1)<=1.0);eUnit(pw(w._1))}).sum 
            val docEnt = mdoc.map(w => eUnit(pw(w._1))).sum 
            (docEnt, mdoc.size)
        }
        mwz.map(x=>docH(x))
    }

    def perplexity(modelInfo:ModelInfo, topicInfo:TopicInfo, data:LDADataRDD) = {
        val tinfo = data.sparkContext broadcast topicInfo
        val minfo = modelInfo
        val (eall, cntall) = data.mapPartitions(it=>LDA.entropySum(minfo, tinfo.value, it.toSeq).toIterator).reduce((x,y)=>(x._1+y._1,x._2+y._2))
        tinfo.unpersist(blocking=true)
        //println(s"#${eall}\t${cntall}")
        val entropy = eall/cntall
        Math.pow(2, entropy)
    }

    def gibbsMapper(modelInfo: ModelInfo, topicInfo:TopicInfo, omwz:Seq[List[(String,Int)]], round:Int, inference:Boolean=false) = {
        import modelInfo._
        val mwz = omwz.map(_.toArray)
        val nzw = mutable.Map(topicInfo.nzw.toSeq:_*).withDefaultValue(0L)
        val nz = new Array[Long](ntopics)
        for((z, c) <- topicInfo.nz) nz(z) = c
        val nmz = genNMZ(omwz, ntopics)
        val probz = new Array[Double](ntopics)
        for(r <- 1 to round) {
            for((mdoc, docmz) <- mwz zip nmz) {
                for(((cw, cz), i) <- mdoc.zipWithIndex) {
                    if (inference == false) {
                        nzw((cz, cw)) -= 1
                        nz(cz) -= 1
                    }
                    docmz(cz) -= 1
                    for(z <- 0 until ntopics) probz(z) = (nzw((z,cw))+beta)/(nz(z)+nterms*beta)*(docmz(z)+alpha)
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

    def wordScores(modelInfo: ModelInfo, topicInfo:TopicInfo, gdocinfo:DocInfo, doc:List[(String, Int)]) = {
        import modelInfo._
        import topicInfo._
        val docinfo = genDocInfo(doc)
        import docinfo._
        def pz(z:Int) = (nmz(z)+alpha)/(nm+ntopics*alpha)
        def pzw(z:Int, w:String) = (nzw((z,w))+beta)/(nz(z)+nterms*beta)
        def pw(w:String) = (0 until ntopics).map(z=>pz(z)*pzw(z,w)).sum
        def gpz(z:Int) = (gdocinfo.nmz(z)+alpha)/(gdocinfo.nm+ntopics*alpha)
        def gpw(w:String) = (0 until ntopics).map(z=>gpz(z)*pzw(z,w)).sum
        def score(w:String) = pw(w)/gpw(w)
        doc.map(_._1).groupBy(x=>x).map(x=>(x._1, x._2.size * score(x._1))).toSeq.sortBy(_._2).reverse.toList
    }
}


class ADLDAModel (
        final val ntopics:Int,
        final val nterms:Int, 
        var data:LDA.LDADataRDD,
        final val npartition:Int=100) 
{

    val minfo = new ModelInfo(ntopics, nterms)
    var tinfo = LDA.genTopicInfo(this.data)
    var perplexity = LDA.perplexity(minfo, tinfo, data)
    var gdocinfo = LDA.genGlobalDocInfo(data)

    def train(round:Int, innerRound:Int = 20) {
        this.data = if (this.data.partitions.size >= this.npartition) this.data else this.data.repartition(this.npartition)
        val minf = this.minfo
        @tailrec def loop(i:Int, mwz:LDA.LDADataRDD, tpinfo:TopicInfo):(LDA.LDADataRDD, TopicInfo) = {
            println(s"Round:\t${i}\nPerplexity:\t${LDA.perplexity(minf, tpinfo, mwz)}")
            if(i == round) (mwz, tpinfo)
            else {
                val tinfo = mwz.sparkContext broadcast tpinfo
                val mwzNew = mwz.mapPartitions(
                        it=>LDA.gibbsMapper(minf, tinfo.value, it.toSeq, innerRound).toIterator, 
                                    preservesPartitioning=true)
                    .persist
                mwzNew.checkpoint
                val tpinfoNew = LDA.genTopicInfo(mwzNew)
                //println(s"@test checkpointed:${mwzNew.isCheckpointed}")
                mwz.unpersist(blocking=true)
                tinfo.unpersist(blocking=true)
                loop(i+1, mwzNew, tpinfoNew)
            }
        }
        val (x, y) = loop(1, this.data, this.tinfo)
        this.data = x
        this.tinfo = y
        this.perplexity = LDA.perplexity(this.minfo, this.tinfo, this.data)
        this.gdocinfo = LDA.genGlobalDocInfo(this.data)
        println(s"Final Perplexity:${this.perplexity}")
    }

    def inference(infer:LDA.LDADataRDD, round:Int = 20, npart:Int = 0) = {
        val todo = if(npart > 0) infer.repartition(npart) else infer
        val minf = this.minfo
        val tinfo = infer.sparkContext broadcast this.tinfo
        infer.mapPartitions(it=>LDA.gibbsMapper(minf, tinfo.value, it.toSeq, round, inference=true).toIterator, 
                                    preservesPartitioning=true)
    }

    def wordScores(doc:List[(String, Int)]) = LDA.wordScores(this.minfo, this.tinfo, this.gdocinfo, doc)

    def wordScores(docs:LDA.LDADataRDD) = {
        val minfo = this.minfo
        val tinfo = this.tinfo
        val gdocinfo = this.gdocinfo
        docs.map(x=>LDA.wordScores(minfo, tinfo, gdocinfo, x))
    }
}
