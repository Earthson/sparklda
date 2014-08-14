package net.earthson.tasks

import net.earthson.{nlp, sparkjob}


object ADLDAJob extends sparkjob.SparkJob {
    override def main(args: Array[String]) {
        val spark = initspark("AD-LDA Testing")
        val ntopics = 128
        val adldaModel = nlp.lda.LDA.loadADLDA(spark, "hdfs://ns1/nlp/lda/wiki.docs.10000", 64)
        adldaModel.train(round=50, innerRound=10)
        for((tp, tpw) <- nlp.lda.LDA.topWords(adldaModel.tinfo)) {
            printf("%d\t:\t%s\n", tp, tpw.take(20).map(_._1).mkString(sep="\t"))
        }
        val scores = adldaModel.wordScores(adldaModel.data)
        nlp.lda.LDA.toFile(adldaModel.data, "hdfs://ns1/nlp/lda/wiki.10000.lda.k64.r500")
        scores.saveAsTextFile("hdfs://ns1/nlp/lda/wiki.wordscores.10000")
        spark.stop()
    }
}
