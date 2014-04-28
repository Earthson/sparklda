package net.earthson.nlp.lda

import scala.collection.mutable
import net.earthson.nlp.sampler.MultiSampler


class ModelInfo(val ntopics:Int, val nterms:Int) {
    val alpha = ntopics/50.0
    val beta = 0.01
}

class TopicInfo(val nzw:Seq[((Int,String),Long)], val nz:Seq[(Int, Long)]) { }

object GibbsMapper {
    def mapper(modelInfo: ModelInfo, topicinfo:TopicInfo, mwz:Seq[(Long,String,Int)], round:Int) = {
        import modelInfo._
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


class LDAModel(var mwz:Seq[(Long,String,Int)], val ntopics:Int) {
    //TODO: Add Perplexity

    val nterms = Set(mwz.map(_._2):_*).size
    val modelinfo = new ModelInfo(ntopics, nterms)

    def nzw = mwz.groupBy(x=>(x._3, x._2)).mapValues(_.size.toLong).toSeq
    def nz = nzw.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum.toLong).toSeq
    def tinfo = new TopicInfo(this.nzw, this.nz)
    
    def train(round:Int) {
        this.mwz = GibbsMapper.mapper(this.modelinfo, this.tinfo, this.mwz, round)
    }

    def twords(limit:Int) = {
        val info = this.nzw.groupBy(_._1._1).map(x=>(x._1,x._2.map(y=>(y._1._2, y._2)).toSeq.sortBy(_._2).reverse)).toSeq.sortBy(_._1)
        for((tp, tpw) <- info) {
            printf("%d\t:%s\n", tp, tpw.take(limit).map(_._1).mkString(sep="\t"))
        }
    }
}

import org.apache.spark.rdd
import org.apache.spark.SparkContext

class ADLDAModel(val sc:SparkContext, var mwz:rdd.RDD[(Long,String,Int)], val ntopics:Int) {
    //TODO: flexible save location and input location
    //TODO: Add Perplexity

    val nterms = mwz.map(_._2).distinct.count.toInt
    val modelinfo = new ModelInfo(ntopics, nterms)

    def nzw = mwz.map(x=>(x._3, x._2)).countByValue.toSeq
    def nz = nzw.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum.toLong).toSeq
    def tinfo = new TopicInfo(this.nzw, this.nz)

    def train(round:Int, inround:Int = 20, savestep:Int = 5) {
        var step = 0
        for(i <- 0 until round) {
            val tmptinfo = this.sc broadcast this.tinfo
            this.mwz = this.mwz.mapPartitions(it=>GibbsMapper.mapper(this.modelinfo, tmptinfo.value, it.toSeq, inround).toIterator)
            step = (step + 1) % savestep
            if (step == 0 && i+1 != round) this.mwz.saveAsTextFile(s"hdfs://ns1/nlp/lda/solution.round.${i}")
        }
        this.mwz.saveAsTextFile(s"hdfs://ns1/nlp/lda/solution.round.final")
    }
}
