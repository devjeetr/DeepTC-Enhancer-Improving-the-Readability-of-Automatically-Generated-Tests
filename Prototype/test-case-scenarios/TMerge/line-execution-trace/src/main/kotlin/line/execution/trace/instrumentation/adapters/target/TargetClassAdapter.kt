package line.execution.trace.instrumentation.adapters.target

import org.objectweb.asm.ClassVisitor
import org.objectweb.asm.MethodVisitor
import org.objectweb.asm.Opcodes

class TargetClassAdapter : ClassVisitor, Opcodes {
    private var source: String = ""

    constructor(cv: ClassVisitor) : super(Opcodes.ASM7, cv) {
    }

    override fun visit(
        version: Int,
        access: Int,
        name: String?,
        signature: String?,
        superName: String?,
        interfaces: Array<out String>?
    ) {
        this.source = name?.replace("/", ".").toString()
        super.visit(version, access, name, signature, superName, interfaces)
    }
    override fun visitSource(source: String?, debug: String?) {
        super.visitSource(source, debug)
    }

    override fun visitMethod(
        access: Int,
        name: String?,
        descriptor: String?,
        signature: String?,
        exceptions: Array<out String>?
    ): MethodVisitor? {
//        println("method: $name")
        return TargetMethodAdapter(
            this.source,
            api = Opcodes.ASM7,
            access = access,
            name = name,
            desc = descriptor,
            mv = cv.visitMethod(access, name, descriptor, signature, exceptions)
        )
    }

    override fun visitEnd() {

        super.visitEnd()
    }
}