package net.earthson.nlp.sampler

import scala.util.Random

object MultiSampler {
    def multisampling(l: Array[Double]) = {
        val r = l.sum * Random.nextFloat
        def loop(i:Int, rd:Double):Int  = {
            i match {
                case 0 => 0
                case x => if (rd <= l(i)) i else loop(i-1, rd-l(i))
            }
        }
        loop(l.size-1, r)
    }
}
