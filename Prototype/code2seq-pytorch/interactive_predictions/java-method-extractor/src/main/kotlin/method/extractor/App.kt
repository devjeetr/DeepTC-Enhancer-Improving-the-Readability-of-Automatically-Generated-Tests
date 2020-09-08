package method.extractor

import com.github.javaparser.StaticJavaParser
import com.github.javaparser.ast.CompilationUnit
import com.github.javaparser.ast.body.MethodDeclaration
import com.github.javaparser.ast.body.VariableDeclarator
import com.github.javaparser.ast.expr.NameExpr
import com.github.javaparser.ast.expr.SimpleName
import com.github.javaparser.ast.visitor.ModifierVisitor
import com.github.javaparser.ast.visitor.Visitable
import com.github.javaparser.ast.visitor.VoidVisitor
import com.github.javaparser.ast.visitor.VoidVisitorAdapter
import com.github.javaparser.printer.YamlPrinter
import com.github.javaparser.symbolsolver.JavaSymbolSolver
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver
import java.nio.file.Paths
import kotlin.streams.toList
import com.xenomachina.argparser.ArgParser

import java.io.File

class MyArgs(parser: ArgParser) {
    val source by parser.positional("source filename")

    val destination by parser.positional("destination folder")
}

fun extract_method(cu: CompilationUnit, methodDeclaration: MethodDeclaration, obfuscate: Boolean=false): CompilationUnit {
    val modifiedCU = cu.clone();
    val methodDeclarations = modifiedCU.findAll(MethodDeclaration::class.java).stream().toList();
    val variableMapping = mutableMapOf<String, String>()
    methodDeclarations.forEach{
        if (it != methodDeclaration) {
            it.parentNode.ifPresent{
                parent -> parent.remove(it)
            }
        } else if (obfuscate) {
            // perform renaming
            val alphabet = "abcdfghijklmnopqrstuvwxyz"
            val variables = it.findAll(VariableDeclarator::class.java).stream().toList()
            variables.forEachIndexed{
                    index, variable ->
                variableMapping[variable.nameAsString] = alphabet[index].toString()
                variable.name.setIdentifier(variableMapping[variable.nameAsString])
                println("Mapping $variable to ${alphabet[index]}")
            }

            it.findAll(NameExpr::class.java).stream().forEach{
                    nameExpr ->
                if(variableMapping.containsKey(nameExpr.nameAsString)) {
                    nameExpr.name.setIdentifier(variableMapping[nameExpr.nameAsString])
                }
            }
        }
    }

    return modifiedCU;
}

fun main(args: Array<String>) {
    ArgParser(args).parseInto(::MyArgs).run {
//        val targetFile = "/media/devjeetroy/ss/testing-project/survey/internal/projects/spring-boot/spring-boot-project/spring-boot/evosuite-tests/org/springframework/boot/json/AbstractJsonParser_ESTest.java"
        val typeSolver = CombinedTypeSolver()
        val reflectionTypeSolver = ReflectionTypeSolver()
        typeSolver.add(reflectionTypeSolver)
        val symbolSolver = JavaSymbolSolver(typeSolver)
        StaticJavaParser.getConfiguration().setSymbolResolver(symbolSolver)

        val cu = StaticJavaParser.parse(Paths.get(source));
        val methodDeclarations = cu.findAll(MethodDeclaration::class.java).stream().toList();
        println(methodDeclarations.size)
        methodDeclarations.forEach{
            method ->
                val methodName = method.name
                val modifiedCU = extract_method(cu.clone(), method)
                val outPath = Paths.get(destination, "$methodName.java")
                outPath.toFile().writeText(modifiedCU.toString())
        }
    }

}