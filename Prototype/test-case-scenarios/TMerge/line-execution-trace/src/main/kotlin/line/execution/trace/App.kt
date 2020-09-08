/*
 * This Kotlin source file was generated by the Gradle 'init' task.
 */
package line.execution.trace

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default
import line.execution.trace.instrumentation.transformers.ScopeTransformer
import line.execution.trace.instrumentation.transformers.TargetTransformer
import mu.KotlinLogging
import java.lang.instrument.Instrumentation

fun main() {
//    sendPacket()
}

class MyArgs(parser: ArgParser) {
    val cti by parser.storing(
        "-c", "--cti", help = "space separated list of classes to instrument"
    )

    val tti by parser.storing(
        "-t", "--test-to-instrument", help="Test for which we register the scope"
    )

    val testsToInclude by parser.storing(
        "-i", help="Test cases to include"
    ).default("")

    val dbLocation by parser.storing(
        "-o", "--output", help="The name of the sqlite database"
    ).default("./data.db")
}

fun parseClassesToInstrument(classesToInstrumentArg: String): Set<String> {
    if(classesToInstrumentArg.isEmpty()) return setOf()

    return classesToInstrumentArg.split(":").map { s -> s.trim() }.toSet()
}

private val logger = KotlinLogging.logger {}
fun premain(agentArgs: String, inst: Instrumentation?) {
    println(agentArgs)
    val arguments = agentArgs.split(" ").toTypedArray()
    ArgParser(arguments).parseInto(::MyArgs).run {
        logger.info("Inside Agent premain")

        val targetClassesToInstrument = parseClassesToInstrument(cti)
        val testClassesToInstrument = parseClassesToInstrument(tti)
        val testsToInclude = parseClassesToInstrument(testsToInclude)

        logger.info("$targetClassesToInstrument")

        LineLogger.initDB(dbLocation)
        LineLogger.setTestClass(tti)

        inst?.addTransformer(
            TargetTransformer(
                targetClassesToInstrument
            )
        )

        inst?.addTransformer(
            ScopeTransformer(
                testClassesToInstrument,
                testsToInclude
            )
        )


    }

   println("Instrumentation finished")
}