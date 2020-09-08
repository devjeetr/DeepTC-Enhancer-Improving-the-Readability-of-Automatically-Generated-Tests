package line.execution.trace.instrumentation.adapters.scope

import org.objectweb.asm.ClassVisitor
import org.objectweb.asm.MethodVisitor
import org.objectweb.asm.Opcodes

class ScopeClassAdapter : ClassVisitor, Opcodes {
    private var source: String = ""
    private val testsToInclude: Set<String>
    constructor(cv: ClassVisitor, testsToInclude: Set<String>) : super(Opcodes.ASM7, cv) {
        this.testsToInclude = testsToInclude
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

    override fun visitMethod(
        access: Int,
        name: String?,
        descriptor: String?,
        signature: String?,
        exceptions: Array<out String>?
    ): MethodVisitor? {
        val includeTest = testsToInclude.isEmpty() || testsToInclude.contains(name)

        return ScopeAdviceAdapter(
            className = this.source,
            enabled=includeTest,
            api = Opcodes.ASM7,
            access = access,
            name = name,
            desc = descriptor,
            mv = cv.visitMethod(access, name, descriptor, signature, exceptions)
        )
    }

}