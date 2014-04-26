package net.earthson.nlp.sampler

import scala.util.Random

object MultiSampler {
    def multisampling(l: List[Double]) = {
        val r = l.sum * Random.nextFloat
        def loop(i:Int, rd:Double, sq:List[Double]):Int  = {
            sq match {
                case Nil => 0
                case h::t => if (rd <= h) i else loop(i+1, rd-h, t)
            }
            }
        loop(0, r, l)
    }
}
