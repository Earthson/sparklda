package net.earthson.nlp


object LocalLDA {
    def main(args: Array[String]) {
        val lines = scala.io.Source.fromFile("/Users/Earthson/Data/wiki/wiki.doc.split.filted.lda.10000").getLines
        val mwz = lines.map(_.split("\t")).map(x=>(x(0).toLong, x(1), x(2).toInt)).toList.toSeq
        val model = new lda.LDALocal(mwz, 64)
        println("Training Begin")
        model.train(20)
        model.twords(20)
    }
}
