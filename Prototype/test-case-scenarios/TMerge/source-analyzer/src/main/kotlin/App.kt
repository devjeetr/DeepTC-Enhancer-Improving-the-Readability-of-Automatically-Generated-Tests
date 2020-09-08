import org.objectweb.asm.ClassReader
import org.objectweb.asm.ClassWriter
import org.objectweb.asm.Opcodes
import org.objectweb.asm.tree.AbstractInsnNode
import org.objectweb.asm.tree.ClassNode
import org.objectweb.asm.tree.FieldInsnNode
import org.objectweb.asm.tree.MethodNode
import java.nio.file.Files
import java.nio.file.Paths

fun getValuesMutatedByMethod(method: MethodNode, fields: Set<String>, owner: String ): List<String> {
    val instructions = method.instructions

    // reference algorithm: https://stackoverflow.com/questions/9634349/detecting-field-mutation-using-asm
    return instructions.asSequence().filter {
        it.type == AbstractInsnNode.FIELD_INSN
    }.map { it as FieldInsnNode }
            .filter { it.opcode == Opcodes.PUTFIELD }
            .filter{it.owner == owner}
            .map { it.name.toString() }
            .filter{fields.contains(it)}
            .distinct().toList()
}

fun getValuesMutatedForAllMethods(classFilePath: String): Map<String, List<String>> {
    val classFile = Files.readAllBytes(Paths.get(classFilePath))
    val reader = ClassReader(classFile)
    val classNode = ClassNode()
    reader.accept(classNode, 0)

    val methods = classNode.methods
    val instanceVariables = classNode.fields.toSet().map{it.name}.toSet()

    return methods.map{ it.name to getValuesMutatedByMethod(it, instanceVariables, classNode.name)}.toMap()
}

fun main() {
    val root = "/home/devjeetroy/Research/Testing-Project/repositories/repos/commons-cli-1.4-src/target/classes/org/apache/commons/cli/OptionGroup.class"
    println(getValuesMutatedForAllMethods(root))
}