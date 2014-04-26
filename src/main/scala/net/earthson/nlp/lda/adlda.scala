package net.earthson.nlp.lda

import scala.collection.mutable
import net.earthson.nlp.sampler.MultiSampler

/*
    val nzw = mutable.Map(mwz.groupBy(x=>(x._3, x._2)).mapValues(_.size).toSeq:_*).withDefaultValue(0)
    val nz = mutable.Map(nzw.toSeq.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum).toSeq:_*).withDefaultValue(0)
    val nmz = mutable.Map(mwz.groupBy(x=>(x._1, x._3)).mapValues(_.size).toSeq:_*).withDefaultValue(0)
    val nm = mutable.Map(nmz.toSeq.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum).toSeq:_*).withDefaultValue(0)
*/


class ModelInfo(val ntopics:Int, val nterms:Int) {
    val alpha = ntopics/50.0
    val beta = 0.01
}

class TopicInfo(val nzw:Seq[((Int,String),Long)], val nz:Seq[(Int, Long)]) { }

object GibbsMapper {
    def mapper(modelInfo: ModelInfo, topicinfo:TopicInfo, mwz:Seq[(Long,String,Int)], round:Int) {
        import modelInfo._
        val mwzbuf = mwz.toBuffer
        val nzw = mutable.Map(topicinfo.nzw:_*).withDefaultValue(0)
        val nz = mutable.Map(topicinfo.nz:_*).withDefaultValue(0)
        val nmz = mutable.Map(mwz.groupBy(x=>(x._1, x._3)).mapValues(_.size).toSeq:_*).withDefaultValue(0)
        val nm = mutable.Map(nmz.toSeq.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum).toSeq:_*).withDefaultValue(0)
        for(r <- 1 to round) {
            //println("Iteration:", r)
            for(i <- 0 until mwzbuf.size) {
                val (cm, cw, cz) = mwzbuf(i)
                nzw((cz, cw)) -= 1
                nz(cz) -= 1
                nmz((cm,cz)) -= 1
                //nm(cm) -= 1
                val probz = for(z <- 0 until ntopics) yield ((nzw((z,cw))+beta)/(nz(z)+nterms*beta))*(nmz((cm,z))+alpha)
                val newz = MultiSampler.multisampling(probz.toList)
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
