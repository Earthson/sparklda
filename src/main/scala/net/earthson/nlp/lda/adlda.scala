package net.earthson.nlp.lda

import scala.collection.mutable
import net.earthson.nlp.util.MultiSampler


class ADLDAModel(k:Int, mwz:Seq[(Int,String,Int)]) {
    var m_w_z = mwz
    val n_z_w = mutable.Map(m_w_z.groupBy(x=>(x._3, x._2)).mapValues(_.size).toSeq:_*).withDefaultValue(0)
    val n_z = mutable.Map(n_z_w.toSeq.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum).toSeq:_*).withDefaultValue(0)
    val n_m_z = mutable.Map(m_w_z.groupBy(x=>(x._1, x._3)).mapValues(_.size).toSeq:_*).withDefaultValue(0)
    val n_m = mutable.Map(n_m_z.toSeq.map(x=>(x._1._1, x._2)).groupBy(_._1).mapValues(_.map(_._2).sum).toSeq:_*).withDefaultValue(0)
    val alpha = n_m.size/50.0
    val beta = 0.01
    val topic_k = k
    val n_term = m_w_z.map(x=>x._2).toSet.size

    def gibbsSampling() {
        val new_nwz = for((g_m, g_w, g_z) <- m_w_z) yield {
            n_z_w((g_z,g_w)) -= 1
            n_z(g_z) -= 1
            n_m_z((g_m,g_z)) -= 1
            n_m(g_m) -= 1
            val prob_z = for(z <- 1 to topic_k) yield ((n_z_w((z,g_w))+beta)/(n_z(z)+n_term*beta))*(n_m_z((g_m,z))+alpha)
            val new_z = MultiSampler.multisampling(prob_z.toList)
            n_z_w((new_z,g_w)) += 1
            n_z(new_z) += 1
            n_m_z((g_m, new_z)) += 1
            n_m(g_m) += 1
            (g_m, g_w, new_z)
        }
        this.m_w_z = new_nwz
    }
}
