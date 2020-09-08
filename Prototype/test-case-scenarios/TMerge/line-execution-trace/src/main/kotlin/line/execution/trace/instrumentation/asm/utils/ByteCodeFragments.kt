package line.execution.trace.instrumentation.asm.utils

import org.objectweb.asm.MethodVisitor
import org.objectweb.asm.Opcodes


fun printLn(text: String, mv: MethodVisitor) {
    mv.visitFieldInsn(Opcodes.GETSTATIC, "java/lang/System",
            "out", "Ljava/io/PrintStream;")
    mv.visitLdcInsn(text)
    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
            "java/io/PrintStream",
            "println",
            "(Ljava/lang/Object;)V", false)
}

fun logLine(filename: String, line: Int, mv: MethodVisitor) {
    mv.visitFieldInsn(Opcodes.GETSTATIC, "line/execution/trace/LineLogger",
            "INSTANCE",
            "Lline/execution/trace/LineLogger;")

    mv.visitLdcInsn(filename)
    mv.visitIntInsn(Opcodes.SIPUSH, line)
    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
            "line/execution/trace/LineLogger",
            "log",
            "(Ljava/lang/String;I)V",
            false)
}

fun logEnterTargetMethod(methodName: String, mv: MethodVisitor) {
    mv.visitFieldInsn(Opcodes.GETSTATIC, "line/execution/trace/LineLogger",
            "INSTANCE",
            "Lline/execution/trace/LineLogger;")

    mv.visitLdcInsn(methodName)
    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
            "line/execution/trace/LineLogger",
            "enterTargetMethod",
            "(Ljava/lang/String;)V",
            false)
}


fun logExitTargetMethod(methodName: String, mv: MethodVisitor) {
    mv.visitFieldInsn(Opcodes.GETSTATIC, "line/execution/trace/LineLogger",
            "INSTANCE",
            "Lline/execution/trace/LineLogger;")

    mv.visitLdcInsn(methodName)
    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
            "line/execution/trace/LineLogger",
            "exitTargetMethod",
            "(Ljava/lang/String;)V",
            false)
}

fun logEnterTestMethod(methodName: String, mv: MethodVisitor) {
    mv.visitFieldInsn(Opcodes.GETSTATIC, "line/execution/trace/LineLogger",
        "INSTANCE",
        "Lline/execution/trace/LineLogger;")

    mv.visitLdcInsn(methodName)
    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
        "line/execution/trace/LineLogger",
        "enterTestMethod",
        "(Ljava/lang/String;)V",
        false)
}

fun logExitTestMethod(methodName: String, mv: MethodVisitor) {
    mv.visitFieldInsn(Opcodes.GETSTATIC, "line/execution/trace/LineLogger",
        "INSTANCE",
        "Lline/execution/trace/LineLogger;")

    mv.visitLdcInsn(methodName)
    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
        "line/execution/trace/LineLogger",
        "exitTestMethod",
        "(Ljava/lang/String;)V",
        false)
}

fun logTestLine(line: Int, mv: MethodVisitor) {
    mv.visitFieldInsn(Opcodes.GETSTATIC, "line/execution/trace/LineLogger",
            "INSTANCE",
            "Lline/execution/trace/LineLogger;")

    mv.visitIntInsn(Opcodes.SIPUSH, line)
    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
            "line/execution/trace/LineLogger",
            "enterTestLine",
            "(I)V",
            false)

}

