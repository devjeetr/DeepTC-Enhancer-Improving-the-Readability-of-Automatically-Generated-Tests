package line.execution.trace.instrumentation.transformers

import line.execution.trace.instrumentation.adapters.target.TargetClassAdapter
import org.objectweb.asm.ClassReader
import org.objectweb.asm.ClassWriter
import java.lang.instrument.ClassFileTransformer
import java.security.ProtectionDomain

class TargetTransformer : ClassFileTransformer {
    private var classesToInstrument: Set<String> = setOf()

    constructor(classesToInstrument: Set<String>) : super() {
        this.classesToInstrument = classesToInstrument
    }

    override fun transform(
        loader: ClassLoader?,
        className: String?,
        classBeingRedefined: Class<*>?,
        protectionDomain: ProtectionDomain?,
        classfileBuffer: ByteArray?
    ): ByteArray? {
        if (classesToInstrument.contains(className?.replace("/", "."))) {
            val reader = ClassReader(classfileBuffer)

            val writer = ClassWriter(reader, ClassWriter.COMPUTE_FRAMES)
            val ca = TargetClassAdapter(writer)

            reader.accept(ca, ClassReader.EXPAND_FRAMES)
            return writer.toByteArray()
        }

        return null
    }
}