package line.execution.trace.instrumentation.transformers

import line.execution.trace.instrumentation.adapters.scope.ScopeClassAdapter
import org.objectweb.asm.ClassReader
import org.objectweb.asm.ClassWriter
import java.lang.instrument.ClassFileTransformer
import java.security.ProtectionDomain

class ScopeTransformer : ClassFileTransformer {
    private val classesToInstrument: Set<String>
    private val testsToInclude: Set<String>

    constructor(classesToInstrument: Set<String>, testsToInclude: Set<String>) : super() {
        this.classesToInstrument = classesToInstrument
        this.testsToInclude = testsToInclude
    }

    override fun transform(
        loader: ClassLoader?,
        className: String?,
        classBeingRedefined: Class<*>?,
        protectionDomain: ProtectionDomain?,
        classfileBuffer: ByteArray?
    ): ByteArray? {
        if (classesToInstrument.contains(className?.replace("/", "."))) {
            println("instrument this")
            println(classesToInstrument)
            try{
                val reader = ClassReader(classfileBuffer)
                val writer = ClassWriter(reader, ClassWriter.COMPUTE_FRAMES)
                val ca = ScopeClassAdapter(
                    writer,
                    testsToInclude
                )

                reader.accept(ca, ClassReader.EXPAND_FRAMES)
                return writer.toByteArray()
            } catch(e: Exception) {
                println("Exception")
                e.printStackTrace()
            }
        }

        return null
    }
}