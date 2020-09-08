package line.execution.trace.instrumentation.adapters.scope

import line.execution.trace.instrumentation.asm.utils.logEnterTestMethod
import line.execution.trace.instrumentation.asm.utils.logExitTestMethod
import line.execution.trace.instrumentation.asm.utils.logTestLine
import org.objectweb.asm.AnnotationVisitor
import org.objectweb.asm.Label
import org.objectweb.asm.MethodVisitor
import org.objectweb.asm.commons.AdviceAdapter

class ScopeAdviceAdapter : AdviceAdapter {
    private val methodName: String
    private val junitAnnotation = "Lorg/junit/Test;"
    private var enabled = false
    private var isTest = false
    private val className: String

    constructor(
        className: String,
        enabled: Boolean,
        api: Int,
        access: Int,
        name: String?,
        desc: String?,
        mv: MethodVisitor
    ) : super(
        api, mv, access,
        name, desc
    ) {
        this.className = className
        this.methodName = name.toString()
        this.enabled = enabled
    }

    override fun visitAnnotation(descriptor: String?, visible: Boolean): AnnotationVisitor? {
        if (descriptor != null) {
            if (descriptor.startsWith(junitAnnotation)) {
                isTest = true
                // we encountered a junit annotaiton
                if (!enabled) {
                    return null
                }
            }
        }

        return super.visitAnnotation(descriptor, visible)
    }

    override fun onMethodEnter() {
        if (isTest && enabled) {
            logEnterTestMethod(methodName, mv)
        }

        super.onMethodEnter()
    }

    override fun visitLineNumber(line: Int, start: Label?) {
        if(isTest) {
            logTestLine(line, mv)
        }
        super.visitLineNumber(line, start)
    }

    override fun onMethodExit(opcode: Int) {
        if (isTest && enabled) {
            logExitTestMethod(methodName, mv)
        }
        super.onMethodExit(opcode)
    }
}