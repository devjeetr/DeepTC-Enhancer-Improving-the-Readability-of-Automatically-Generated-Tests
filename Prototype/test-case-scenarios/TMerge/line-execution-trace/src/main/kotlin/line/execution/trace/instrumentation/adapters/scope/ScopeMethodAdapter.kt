package line.execution.trace.instrumentation.adapters.scope

import line.execution.trace.instrumentation.asm.utils.logEnterTestMethod
import mu.KotlinLogging
import org.objectweb.asm.AnnotationVisitor
import org.objectweb.asm.MethodVisitor
import org.objectweb.asm.Opcodes

private val logger = KotlinLogging.logger {}
const val junitAnnotation = "Lorg/junit/Test;"
class ScopeMethodAdapter : MethodVisitor, Opcodes {
    private val filename: String
    private val methodName: String
    private val includeTest: Boolean
    private var isTest: Boolean = false

    constructor(filename: String, methodName: String, includeTest: Boolean, mv: MethodVisitor?) : super(Opcodes.ASM7, mv) {
        this.filename = filename
        this.methodName = methodName
        this.includeTest = includeTest
    }


    override fun visitAnnotation(descriptor: String?, visible: Boolean): AnnotationVisitor? {
        if (!includeTest && descriptor.equals(junitAnnotation)) {
            return null;
        }
        if(descriptor.equals(junitAnnotation)) {
            isTest = true
        }
        return super.visitAnnotation(descriptor, visible)
    }
    override fun visitCode() {
        if (isTest && includeTest) {
            logEnterTestMethod(methodName, mv)
        }
        super.visitCode()

    }


}