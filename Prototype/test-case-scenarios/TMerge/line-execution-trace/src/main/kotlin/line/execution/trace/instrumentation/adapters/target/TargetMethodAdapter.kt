package line.execution.trace.instrumentation.adapters.target

import line.execution.trace.instrumentation.asm.utils.*
import org.objectweb.asm.AnnotationVisitor
import org.objectweb.asm.Label
import org.objectweb.asm.MethodVisitor
import org.objectweb.asm.Opcodes
import org.objectweb.asm.commons.AdviceAdapter

//class TargetMethodAdapter : MethodVisitor, Opcodes {
//    private var filename: String?
//
//    constructor(filename: String, mv: MethodVisitor?) : super(Opcodes.ASM7, mv) {
//        this.filename = filename
//    }
//
//    override fun visitMethodInsn(
//        opcode: Int,
//        owner: String?,
//        name: String?,
//        descriptor: String?,
//        isInterface: Boolean
//    ) {
//        super.visitMethodInsn(opcode, owner, name, descriptor, isInterface)
//    }
//
//    override fun visitLineNumber(line: Int, start: Label?) {
//        mv.visitLineNumber(line, start)
//        logLine(filename.toString(), line, mv)
//    }
//
//
//    override fun visitFieldInsn(opcode: Int, owner: String?, name: String?, descriptor: String?) {
//        mv.visitFieldInsn(opcode, owner, name, descriptor)
//    }
//}


class TargetMethodAdapter(private val className: String, api: Int, access: Int, name: String?, desc: String?, mv: MethodVisitor) : AdviceAdapter(
        api, mv, access,
        name, desc
) {
    override fun visitLineNumber(line: Int, start: Label?) {

        super.visitLineNumber(line, start)
        logLine(className, line, mv)
    }
}
